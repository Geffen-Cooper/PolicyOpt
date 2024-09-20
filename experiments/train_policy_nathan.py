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
	device = "cpu"
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
	BUFFER_SIZE = 5
	IN_DIM = 2*BUFFER_SIZE
	HIDDEN_DIM = 32
	BATCH_SIZE = 32
	LR = 1e-3
	MOMENTUM = 0.9
	WD = 1e-4
	ITERATIONS = 5000

	DURATION = 30 # length of segments in seconds
	FS = 25 # sampling frequency
	PACKET_SIZE = 8
	LEAKAGE = 6e-6
	INIT_OVERHEAD = 150e-6

	eh = EnergyHarvester()

	policy = EnergyPolicy(IN_DIM,HIDDEN_DIM).to(device)
	opt = torch.optim.SGD(policy.parameters(),lr=LR,momentum=MOMENTUM,weight_decay=WD)
	loss_fn_har = torch.nn.CrossEntropyLoss()
	loss_fn_policy = torch.nn.BCELoss()

	for iteration in range(ITERATIONS):
		Loss = 0
		opp_batch_loss = 0
		learned_batch_loss = 0
		policy_outputs = torch.zeros(BUFFER_SIZE, dtype=torch.float32, requires_grad=True, device=device)
		actions = torch.zeros(BUFFER_SIZE, dtype=torch.float32, requires_grad=True, device=device)
		for batch_idx in range(BATCH_SIZE):
			policy_outputs = policy_outputs[-BUFFER_SIZE:]
			actions = actions[-BUFFER_SIZE:]
			segment_data, segment_labels = sample_segment(DURATION*FS, train_data, train_labels) # what are train_labels? A: activity labels

			# add time axis
			t_axis = np.arange(len(segment_labels))/FS
			t_axis = np.expand_dims(t_axis,axis=0).T

			# add the time axis to the data
			full_data_window = np.concatenate([t_axis,segment_data],axis=1)

			# opp_packets, opp_e_trace = sparsify_data(full_data_window,PACKET_SIZE,LEAKAGE,INIT_OVERHEAD,eh,'opportunistic',train_mode=False)
			
            # this is where the constraints of the problem are....
			learned_packets, learned_e_trace, policy_outputs, actions = sparsify_data(
				full_data_window,
				PACKET_SIZE,LEAKAGE,INIT_OVERHEAD,eh,'learned_policy',policy,train_mode=True, 
				past_policy_outputs=policy_outputs, past_actions=actions, device=device, history_size=BUFFER_SIZE)

			# if len(opp_packets[0]) == 0 or len(learned_packets[0]) == 0:
			# 	print(opp_packets[0],learned_packets[0])
			# 	continue

			# dense_outputs, dense_preds, dense_targets, dense_outputs_opp, dense_preds_opp, dense_targets_policy = classify_packets(segment_data,segment_labels,opp_packets,model,PACKET_SIZE)
			# exit()
			dense_outputs, dense_preds, dense_targets, dense_outputs_learned, dense_preds_learned, dense_targets_policy = classify_packets(segment_data,segment_labels,learned_packets,model,PACKET_SIZE)

			# fake_labels = (torch.cat(policy_outputs) > 0.5).float()
			try:
				fake_labels = torch.cat(actions).float()
			except:
				continue

			# classification loss
			# opp_classification_loss = loss_fn_har(dense_outputs_opp,dense_targets.long())
			learned_classification_loss = loss_fn_har(dense_outputs_learned, dense_targets)

			print(dense_outputs_learned.shape, dense_targets.shape)

			# policy loss (does mean by default)
			policy_loss = loss_fn_policy(policy_outputs,actions)

			# modulate loss by comparing policies
			# Case 1: learned policy results in lower classification loss
			#	-then reference is (+) and we reinforce current decisions
			# Case 2: learned policy results in higher classification loss
			#	-then reference is (-) and we flip sign of gradient to discourage decisions
			# reference =  opp_classification_loss - learned_classification_loss
			# print(f"iteration: {iteration}, batch_idx: {batch_idx}, opp_cl_loss: {opp_classification_loss}, l_cl_loss: {learned_classification_loss}")

			# print(opp_classification_loss, learned_classification_loss, policy_loss)
			# print(dense_outputs_opp.shape,dense_targets.shape)
			# print(F.softmax(dense_outputs_opp,dim=1).shape)
			# print(F.softmax(dense_outputs_opp,dim=1)[:,int(dense_targets[0])], dense_targets)
			# exit()

			# Loss += 1/BATCH_SIZE*reference*policy_loss
			Loss += 1/BATCH_SIZE * learned_classification_loss

			print('Policy loss', policy_loss)
			print('Classification loss', learned_classification_loss)
			print('Loss', Loss)

			# opp_batch_loss += opp_classification_loss
			learned_batch_loss += learned_classification_loss

		# writer.add_scalar(f"c_loss/diff", opp_batch_loss/BATCH_SIZE-learned_batch_loss/BATCH_SIZE, iteration)
		writer.add_scalar(f"c_loss/learned", learned_batch_loss/BATCH_SIZE, iteration)

		opt.zero_grad()
		Loss.backward()
		opt.step()

		writer.add_scalar(f"train_metric/batch_loss", Loss, iteration)

		print(f"Iteration: {iteration}, batch loss: {Loss}")
