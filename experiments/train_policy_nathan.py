import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.setup_funcs import *
from datasets.dsads_contig.dsads import *
from train import *
from models import *
from datasets.apply_policy_nathan import *


def sample_segment(duration, data, labels):
	rand_start = int(np.random.rand()*len(labels))

	# make sure segment doesn't exceed end of data
	if rand_start + duration >= len(labels):
		rand_start = len(labels) - duration

	data_seg = data[rand_start:rand_start+duration,:]
	label_seg = labels[rand_start:rand_start+duration]

	return data_seg, label_seg
	

if __name__ == '__main__':
	device = "cuda" if torch.cuda.is_available() else "cpu"
	# start tensorboard session
	writer = SummaryWriter(os.path.join(PROJECT_ROOT,"saved_data/runs","policy")+"_"+str(time.time()))

	# load data
	root_dir = os.path.join(PROJECT_ROOT,"datasets/dsads_contig/merged_preprocess")
	train_data = np.load(f"{root_dir}/training_data.npy")
	train_labels = np.load(f"{root_dir}/training_labels.npy")

	val_data = np.load(f"{root_dir}/val_data.npy")
	val_labels = np.load(f"{root_dir}/val_labels.npy")

	test_data = np.load(f"{root_dir}/testing_data.npy")
	test_labels = np.load(f"{root_dir}/testing_labels.npy")

	# load pretrained classifier
	model = SimpleNet(3,10).to(device)
	ckpt_path = os.path.join(PROJECT_ROOT,f"saved_data/checkpoints/seed{123}.pth")
	model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

	# hyperparameters
	BUFFER_SIZE = 20 # TODO: there may be an error when buffer size is not 20 
	IN_DIM = 2*BUFFER_SIZE
	HIDDEN_DIM = 32
	BATCH_SIZE = 32
	LR = 1e-3
	MOMENTUM = 0.9
	WD = 1e-4
	ITERATIONS = 5000
	VAL_EVERY_ITERATIONS = 10

	DURATION = 30 # length of segments in seconds
	FS = 25 # sampling frequency
	PACKET_SIZE = 8
	LEAKAGE = 6e-6
	INIT_OVERHEAD = 150e-6

	assert(PACKET_SIZE <= BUFFER_SIZE), f"Packet size must be smaller than buffer size. Got packet size {PACKET_SIZE} and buffer size {BUFFER_SIZE}"

	eh = EnergyHarvester()

	policy = EnergyPolicy(IN_DIM,HIDDEN_DIM).to(device)
	opt = torch.optim.SGD(policy.parameters(),lr=LR,momentum=MOMENTUM,weight_decay=WD)
	loss_fn_har = torch.nn.CrossEntropyLoss()
	# loss_fn_policy = torch.nn.BCELoss()

	best_val_f1 = 0.0

	for iteration in tqdm(range(ITERATIONS)):
		Loss = 0
		opp_batch_loss = 0
		learned_batch_loss = 0
		for batch_idx in range(BATCH_SIZE):
			train_segment_data, train_segment_labels = sample_segment(DURATION*FS, train_data, train_labels)
			train_segment_labels = torch.tensor(train_segment_labels, dtype=torch.long, device=device)

			# add time axis
			train_t_axis = np.arange(len(train_segment_labels))/FS
			train_t_axis = np.expand_dims(train_t_axis,axis=0).T

			# add the time axis to the data
			train_full_data_window = np.concatenate([train_t_axis,train_segment_data],axis=1)
			
			learned_packets, learned_e_trace, policy_outputs, actions = sparsify_data(
				train_full_data_window,
				PACKET_SIZE,LEAKAGE,INIT_OVERHEAD,eh,'learned_policy',policy,train_mode=True, 
				device=device, history_size=BUFFER_SIZE, sample_frequency=FS)
			
			if len(learned_packets[0]) == 0:
				# Policy did not sample at all
				continue

			dense_outputs, dense_preds, dense_targets, dense_outputs_learned, dense_preds_learned, dense_targets_policy = classify_packets(train_segment_data,train_segment_labels,learned_packets,model,PACKET_SIZE, device=device)

			# classification loss
			learned_classification_loss = loss_fn_har(dense_outputs, dense_targets)
			Loss += 1/BATCH_SIZE * learned_classification_loss
			learned_batch_loss += learned_classification_loss
			# learned_classification_loss.register_hook(lambda grad: print('Gradient:', grad))

		opt.zero_grad()
		Loss.backward()
		opt.step()

		train_f1 = f1_score(
			dense_targets.detach().cpu().numpy(), dense_preds.detach().cpu().numpy(), average='macro'
			)
		
		print(f"Iteration: {iteration}, batch loss: {Loss}, train f1 score: {train_f1}")

		writer.add_scalar(f"train_metric/batch_loss", Loss, iteration)
		writer.add_scalar(f"train_metric/f1", train_f1, iteration)
		
		# validation
		if iteration != 0 and iteration % VAL_EVERY_ITERATIONS == 0:
			VAL_ITERATIONS = 5
			val_f1 = 0.0
			val_loss = 0.0
			for e in range(VAL_ITERATIONS):

				with torch.no_grad():
					val_segment_data, val_segment_labels = sample_segment(DURATION*FS, val_data, val_labels)
					val_t_axis = np.arange(len(val_segment_labels))/FS
					val_t_axis = np.expand_dims(val_t_axis,axis=0).T
					val_full_data_window = np.concatenate([val_t_axis,val_segment_data],axis=1)

					learned_packets, learned_e_trace, policy_outputs, actions = sparsify_data(
						train_full_data_window,
						PACKET_SIZE,LEAKAGE,INIT_OVERHEAD,eh,'learned_policy',policy,train_mode=True, 
						device=device, history_size=BUFFER_SIZE, sample_frequency=FS)
					
					if len(learned_packets[0]) == 0:
						# Policy did not sample at all
						continue

					dense_outputs, dense_preds, dense_targets, dense_outputs_learned, dense_preds_learned, dense_targets_policy = classify_packets(val_segment_data,val_segment_labels,learned_packets,model,PACKET_SIZE, device=device)

					learned_classification_loss = loss_fn_har(dense_outputs, dense_targets)

					val_loss += learned_classification_loss
					val_f1 += f1_score(
						dense_targets.detach().cpu().numpy(), dense_preds.detach().cpu().numpy(), average='macro'
					)

			val_loss = val_loss / VAL_ITERATIONS
			val_f1 = val_f1 / VAL_ITERATIONS

			print(f"Iteration: {iteration}, val_f1_score: {val_f1}")

			writer.add_scalar(f"val_metric/batch_loss", val_loss, iteration)
			writer.add_scalar(f"val_metric/f1", val_f1, iteration)

			if val_f1 > best_val_f1:
				best_val_f1 = val_f1
				torch.save({
					'epoch': iteration + 1,
					'model_state_dict': model.state_dict(),
					'val_f1': val_f1,
					'val_loss': val_loss,
				}, ckpt_path)	
