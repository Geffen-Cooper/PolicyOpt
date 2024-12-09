import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pickle
import argparse

from utils.setup_funcs import *
from datasets.dsads_contig.dsads import *
from train import *
from models import *
from datasets.apply_policy import *

def gen_test_seq(data,labels,min_duration,max_duration,og_sampling_rate, seed):
		np.random.seed(seed)
		activity_idxs = {i : (labels == i).nonzero()[0] for i in np.unique(labels)}
		# print(activity_idxs)
		duration = np.arange(min_duration,max_duration+1)

		X = np.zeros_like(data)
		y = np.zeros_like(labels)

		activity_counters = np.zeros(len(np.unique(labels))) # keeps track of where we are
		remaining_activities = list(np.unique(labels))
		sample_counter = 0

		while len(remaining_activities) > 0:
			# randomly sample an activity
			act = int(np.random.choice(np.array(remaining_activities), 1)[0])

			# randomly sample a duration
			dur = np.random.choice(duration, 1)[0]

			# access this chunk of data and add to sequence
			# print(act)
			start = int(activity_counters[act])
			end = int(start + dur*og_sampling_rate)

			activity_counters[act] += (end-start)

			# check if hit end
			if end >= activity_idxs[act].shape[0]:
				end = int(activity_idxs[act].shape[0])-1
				remaining_activities.remove(act)
				print(remaining_activities)
			# print(activity_counters)
			# print(start,end)
			start = activity_idxs[act][start]
			end = activity_idxs[act][end]
			# print(start,end)
			X[sample_counter:sample_counter+end-start,:] = data[start:end,:]
			y[sample_counter:sample_counter+end-start] = labels[start:end]
			sample_counter += (end-start)

		return X,y

def error_decomp(dense_labels, dense_preds, packets):
	# first pad preds to be full time range so packet idxs are valid
	dense_preds = np.concatenate([np.zeros(len(dense_labels)-len(dense_preds)),dense_preds])

	# get the activity transition points (include starting point idx 0)
	activity_transition_idxs = np.concatenate([np.array([0]),np.where(dense_labels[:-1] != dense_labels[1:])[0] + 1])

	# get the packet arrival points
	packet_arrival_idxs = (packets[0]*25).astype(int)

	# get errors
	classif_total = 0
	classif_correct = 0
	transition_total = 0
	transition_correct = 0


	for i in range(-1,len(packet_arrival_idxs)-1):
		if i == -1:
			curr_packet_idx = 0 
		else:
			curr_packet_idx = packet_arrival_idxs[i]
		next_packet_idx = packet_arrival_idxs[i+1]

		# determine of there is a transition between the packets
		idxs_between = activity_transition_idxs[(activity_transition_idxs > curr_packet_idx) & (activity_transition_idxs < next_packet_idx)]
		if len(idxs_between) > 0:
			# some transition error
			# everything up until the first transition is classification error
			# everything after the first transition is transition error
			classif_total += (idxs_between[0] - curr_packet_idx)
			classif_correct += np.array((dense_labels[curr_packet_idx:idxs_between[0]] == dense_preds[curr_packet_idx:idxs_between[0]])).sum()
			transition_total += (next_packet_idx - idxs_between[0])
			transition_correct += np.array((dense_labels[idxs_between[0]:next_packet_idx] == dense_preds[idxs_between[0]:next_packet_idx])).sum()
		else:
			# all classif error
			classif_total += (next_packet_idx - curr_packet_idx)
			classif_correct += np.array((dense_labels[curr_packet_idx:next_packet_idx] == dense_preds[curr_packet_idx:next_packet_idx])).sum()
	
	# after the last packet is sent, there some remaining region left
	curr_packet_idx = next_packet_idx
	next_packet_idx = len(dense_labels)

	# determine of there is a transition between the packets
	idxs_between = activity_transition_idxs[(activity_transition_idxs > curr_packet_idx) & (activity_transition_idxs < next_packet_idx)]
	if len(idxs_between) > 0:
		# some transition error
		# everything up until the first transition is classification error
		# everything after the first transition is transition error
		classif_total += (idxs_between[0] - curr_packet_idx)
		classif_correct += np.array((dense_labels[curr_packet_idx:idxs_between[0]] == dense_preds[curr_packet_idx:idxs_between[0]])).sum()
		transition_total += (next_packet_idx - idxs_between[0])
		transition_correct += np.array((dense_labels[idxs_between[0]:next_packet_idx] == dense_preds[idxs_between[0]:next_packet_idx])).sum()
	else:
		# all classif error
		classif_total += (next_packet_idx - curr_packet_idx)
		classif_correct += np.array((dense_labels[curr_packet_idx:next_packet_idx] == dense_preds[curr_packet_idx:next_packet_idx])).sum()


	print(f"Total Preds: {len(dense_labels)}")
	print(f"Accuracy: {np.array(dense_preds == dense_labels).mean()} ({np.array(dense_preds == dense_labels).sum()/len(dense_labels)})")
	print(f"Classif Frac: {classif_total}/{len(dense_labels)} ({classif_total/len(dense_labels)}) -- {classif_correct} correct, {classif_total-classif_correct} incorrect")
	print(f"Transition Frac: {transition_total}/{len(dense_labels)} ({transition_total/len(dense_labels)}) -- {transition_correct} correct, {transition_total-transition_correct} incorrect")
	exit()
	return classif_total, classif_correct, transition_total, transition_correct


if __name__ == '__main__':
	# ============ Argument parser ============
	parser = argparse.ArgumentParser(description='Sequence Generation')
	parser.add_argument('--root_dir', type=str, default="~/Projects/data/dsads", help='path to dsads dataset')
	parser.add_argument('--seed', type=int, default=0, help='random seed')
	parser.add_argument('--min_duration', type=int, default=10, help='minimum activity duration')
	parser.add_argument('--max_duration', type=int, default=30, help='maximum activity duration')

	args = parser.parse_args()

	# load data
	root_dir = os.path.join(args.root_dir,"merged_preprocess")
	train_data = np.load(f"{root_dir}/training_data.npy")
	train_labels = np.load(f"{root_dir}/training_labels.npy")

	val_data = np.load(f"{root_dir}/val_data.npy")
	val_labels = np.load(f"{root_dir}/val_labels.npy")

	test_data = np.load(f"{root_dir}/testing_data.npy")
	test_labels = np.load(f"{root_dir}/testing_labels.npy")

	test_data, test_labels = gen_test_seq(test_data, test_labels, args.min_duration, args.max_duration, 25, args.seed)

	# preprocess
	mean = torch.tensor(np.mean(train_data,axis=0))
	std = torch.tensor(np.std(train_data,axis=0))
	# cannot apply here because need units of m/s^2 for harvesting
	# train_data = (train_data-mean)/(std + 1e-5)
	# val_data = (val_data-mean)/(std + 1e-5)
	# test_data = (test_data-mean)/(std + 1e-5)

	activities = np.load("../datasets/dsads_contig/activity_list.npy")

	# load pretrained classifier
	model = SimpleNet(3,len(np.unique(test_labels)))
	ckpt_path = os.path.join(PROJECT_ROOT,f"saved_data/checkpoints/dsads_contig/seed{123}_activities_{activities}.pth")
	model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

	FS = 25 # sampling frequency
	PACKET_SIZE = 8
	LEAKAGE = 6e-6
	INIT_OVERHEAD = 150e-6
	
	# val_data = val_data[0:2000,:]
	# val_labels = val_labels[0:2000]

	eh = EnergyHarvester()
	
	# add time axis
	t_axis = np.arange(len(test_labels))/FS
	t_axis = np.expand_dims(t_axis,axis=0).T
	
	# add the time axis to the data
	full_data_window = np.concatenate([t_axis,test_data],axis=1)

	opp_packets, opp_e_trace = sparsify_data(full_data_window,PACKET_SIZE,LEAKAGE,INIT_OVERHEAD,eh,'opportunistic',train_mode=False)
	np.save("opp_trace.npy",opp_e_trace)
			
	dense_outputs, dense_preds, dense_targets, dense_outputs_opp, dense_preds_opp, dense_targets_policy = classify_packets(test_data,test_labels,opp_packets,model,PACKET_SIZE, mean, std)

	opp_acc = f1_score(dense_targets, dense_preds_opp,average='macro')
	dense_acc = f1_score(dense_targets, dense_preds,average='macro')

	print(f"Opp F1: {opp_acc}\nDense F1: {dense_acc}")
	with open('targets.pkl', 'wb') as file:
		# Serialize the object and save it to the file
		pickle.dump(dense_targets, file)
	with open('preds.pkl', 'wb') as file:
		# Serialize the object and save it to the file
		pickle.dump(dense_preds_opp, file)
	with open('packets.pkl', 'wb') as file:
		# Serialize the object and save it to the file
		pickle.dump(opp_packets, file)
	error_decomp(test_labels,dense_preds_opp,opp_packets)