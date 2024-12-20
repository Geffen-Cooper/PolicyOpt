import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pickle
import argparse

from utils.setup_funcs import *
from datasets.dsads_contig.dsads import *
from train import *
from models import *
from datasets.apply_policy import *
from sklearn.metrics import recall_score

def get_recall(model):
	_,_,test_loader = load_dsads_person_dataset(128)
	model.eval()

	with torch.no_grad():
		predictions = []
		labels = []
		outputs = []

		for batch_idx, (data,target) in enumerate(test_loader):

			# get model output
			out = model(data.float())

			# parse the output for the prediction
			prediction = out.argmax(dim=1).to('cpu')

			predictions.append(prediction)
			labels.append(target.to('cpu'))
		
		predictions = torch.cat(predictions).numpy()
		labels = torch.cat(labels).numpy()

		recall = recall_score(labels,predictions,average=None)
		return recall

def get_avg_duration(test_sequence_labels):
	labels = np.unique(test_sequence_labels)
	durations = np.zeros(len(labels))
	for label in labels:
		idxs = (test_sequence_labels == label).nonzero()[0]
		
		change_idxs = (idxs[1:]-idxs[:-1] > 1).nonzero()[0]
		change_idxs = np.concatenate([np.array([-1]),change_idxs,np.array([len(idxs)-1])])
		segment_lens = np.diff(change_idxs)
		# print(segment_lens)
		avg_len = np.mean(segment_lens)
		durations[int(label)] = avg_len/25 # in seconds

	return durations

def get_avg_send_time(test_sequence_labels, packets):
	labels = np.unique(test_sequence_labels)
	durations = np.zeros(len(labels))

	# get the packet arrival points
	packet_arrival_idxs = (packets[0]*25).astype(int)

	for label in labels:
		idxs = (test_sequence_labels == label).nonzero()[0]
		change_idxs = (idxs[1:]-idxs[:-1] > 1).nonzero()[0]
		change_idxs = np.concatenate([np.array([-1]),change_idxs,np.array([len(idxs)-1])])
		segment_lens = np.diff(change_idxs)
		
		change_idxs[-1] -= 1
		starts = idxs[change_idxs+1][:-1]
		ends = starts + segment_lens

		t_send_avg = 0
		for start,end in zip(starts,ends):
			idxs_between = packet_arrival_idxs[(packet_arrival_idxs > start) & (packet_arrival_idxs < end)]
			if len(idxs_between) > 0:
				t_send_avg += (idxs_between[0]-start)
			else:
				t_send_avg += end-start

		durations[int(label)] = (t_send_avg/25)/len(starts)

	return durations



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


	total_preds = len(dense_labels)
	opp_accuracy = np.array(dense_preds == dense_labels).mean()
	active_region = classif_total/total_preds
	passive_region = transition_total/total_preds
	active_error = (classif_total-classif_correct)/total_preds
	passive_error = (transition_total-transition_correct)/total_preds
	print(f"Total Preds: {total_preds}")
	print(f"Accuracy: {opp_accuracy}")
	print(f"Active Region Frac. Frac: {classif_total}/{total_preds} ({active_region}) -- {classif_correct} correct, {classif_total-classif_correct} incorrect")
	print(f"Passive Region Frac: {transition_total}/{total_preds} ({passive_region}) -- {transition_correct} correct, {transition_total-transition_correct} incorrect")
	print(f"Error: {1-opp_accuracy}, Active: {active_error}, Passive: {passive_error}")
	return opp_accuracy, active_error, passive_error

if __name__ == '__main__':
	# ============ Argument parser ============
	parser = argparse.ArgumentParser(description='Sequence Generation')
	parser.add_argument('--root_dir', type=str, default="~/Projects/data/dsads", help='path to dsads dataset')
	parser.add_argument('--seed', type=int, default=0, help='random seed')
	# parser.add_argument('--min_duration', type=int, default=10, help='minimum activity duration')
	# parser.add_argument('--max_duration', type=int, default=30, help='maximum activity duration')
	parser.add_argument('--duration_list', type=int, nargs='+', default=[10,30], help='durations to sweep through')

	args = parser.parse_args()

	activities = np.load("../datasets/dsads_contig/activity_list.npy")

	# load data
	root_dir = os.path.join(args.root_dir,"merged_preprocess")
	train_data = np.load(f"{root_dir}/training_data.npy")
	train_labels = np.load(f"{root_dir}/training_labels.npy")

	val_data = np.load(f"{root_dir}/val_data.npy")
	val_labels = np.load(f"{root_dir}/val_labels.npy")

	test_data = np.load(f"{root_dir}/testing_data.npy")
	test_labels = np.load(f"{root_dir}/testing_labels.npy")

	results_table = {}

	for min_dur_i, min_dur in enumerate(args.duration_list[:-1]):
		for max_dur in args.duration_list[min_dur_i+1:]:
			results_table[(min_dur,max_dur)] = np.zeros((2,3,4)) # 2 policies, 3 seeds, 4 values
			for seed in np.arange(3):
				print(f"duration: [{min_dur},{max_dur}], seed: {seed}")

				test_data_seq, test_labels_seq = gen_test_seq(test_data, test_labels, min_dur, max_dur, 25, seed)

				
				# preprocess
				mean = torch.tensor(np.mean(train_data,axis=0))
				std = torch.tensor(np.std(train_data,axis=0))
				# cannot apply here because need units of m/s^2 for harvesting
				# train_data = (train_data-mean)/(std + 1e-5)
				# val_data = (val_data-mean)/(std + 1e-5)
				# test_data = (test_data-mean)/(std + 1e-5)

				

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
				t_axis = np.arange(len(test_labels_seq))/FS
				t_axis = np.expand_dims(t_axis,axis=0).T
				
				# add the time axis to the data
				full_data_window = np.concatenate([t_axis,test_data_seq],axis=1)

				opp_packets, opp_e_trace = sparsify_data(full_data_window,PACKET_SIZE,LEAKAGE,INIT_OVERHEAD,eh,'opportunistic',train_mode=False)
				
				t_send = get_avg_send_time(test_labels_seq,opp_packets)
				t_dur = get_avg_duration(test_labels_seq)
				eps = 1-get_recall(model)
				is_active = t_send < 0.5*t_dur

				orac_packets, orac_e_trace = sparsify_data(full_data_window,PACKET_SIZE,LEAKAGE,INIT_OVERHEAD,eh,'oracle_0.2',train_mode=False,label_sequence=test_labels_seq,is_active=is_active)
				# exit()
				# continue		
				dense_outputs, dense_preds, dense_targets, dense_outputs_opp, dense_preds_opp, dense_targets_policy = classify_packets(test_data_seq,test_labels_seq,opp_packets,model,PACKET_SIZE, mean, std)
				_, _, dense_targets_orac, _, dense_preds_orac, _ = classify_packets(test_data_seq,test_labels_seq,orac_packets,model,PACKET_SIZE, mean, std)
				# print(len(opp_e_trace))
				# plt.plot(opp_e_trace*10e6/100)
				# plt.plot(np.concatenate([np.zeros(len(opp_e_trace)-len(dense_targets)),dense_targets]),label="targets")
				# plt.plot(np.concatenate([np.zeros(len(opp_e_trace)-len(dense_targets)),dense_preds]),label="dense")
				# plt.plot(np.concatenate([np.zeros(len(opp_e_trace)-len(dense_targets)),dense_preds_opp]),label="opp")
				# plt.legend()
				# plt.show()
				opp_acc = f1_score(dense_targets, dense_preds_opp,average='macro')
				orac_acc = f1_score(dense_targets_orac, dense_preds_orac,average='macro')
				dense_acc = f1_score(dense_targets, dense_preds,average='macro')

				dense_accuracy = np.array(dense_targets == dense_preds).mean()

				print(f"Opp F1: {opp_acc}\nOrac F1: {orac_acc}\nDense F1: {dense_acc}")
				# with open('targets.pkl', 'wb') as file:
				# 	# Serialize the object and save it to the file
				# 	pickle.dump(dense_targets, file)
				# with open('preds.pkl', 'wb') as file:
				# 	# Serialize the object and save it to the file
				# 	pickle.dump(dense_preds_opp, file)
				# with open('packets.pkl', 'wb') as file:
				# 	# Serialize the object and save it to the file
				# 	pickle.dump(opp_packets, file)
				opp_accuracy, active_error, passive_error = error_decomp(test_labels_seq,dense_preds_opp,opp_packets)
				orac_accuracy, active_error_o, passive_error_o = error_decomp(test_labels_seq,dense_preds_orac,orac_packets)
				print(f"Opp Acc: {opp_acc}, Opp Aerr: {active_error}, Opp Perr: {passive_error}")
				print(f"Orac Acc: {orac_acc}, Orac Aerr: {active_error_o}, Orac Perr: {passive_error_o}")
				# exit()
				results_table[(min_dur,max_dur)][0][seed] = np.array([dense_accuracy, opp_accuracy, active_error, passive_error])
				results_table[(min_dur,max_dur)][1][seed] = np.array([dense_accuracy, orac_accuracy, active_error_o, passive_error_o])
	# exit()
	with open('results.pkl', 'wb') as file:
		# Serialize the object and save it to the file
		pickle.dump(results_table, file)

	with open('results.pkl', 'rb') as file:
		results_table = pickle.load(file)


	fig,ax = plt.subplots(1,1,figsize=(12,5))
	legend_plotted = False
	blank_labels = ['','','','']
	legend_labels = ['Dense Accuracy','Opportunistic Accuracy','Active Error','Passive Error']
	legend_labels_orac = ['Dense Accuracy','Oracle Accuracy','Active Error Oracle','Passive Error Oracle']
	for min_dur_i, min_dur in enumerate(args.duration_list[:-1]):
		for max_dur in args.duration_list[min_dur_i+1:]:
			mean_opp = results_table[(min_dur,max_dur)][0].mean(axis=0)
			std_opp = results_table[(min_dur,max_dur)][0].std(axis=0)
			mean_orac = results_table[(min_dur,max_dur)][1].mean(axis=0)
			std_orac = results_table[(min_dur,max_dur)][1].std(axis=0)
			# print([(max_dur+min_dur)/2],[mean])
			if legend_plotted is False:
				labels = legend_labels
				labels_o = legend_labels_orac
			else:
				labels = blank_labels
				labels_o = blank_labels

			ax.errorbar([(max_dur+min_dur)/2],[mean_opp[0]],yerr=std_opp[0],marker='o',capsize=5,capthick=2, c='k',label=labels[0])

			ax.errorbar([(max_dur+min_dur)/2],[mean_opp[1]],yerr=std_opp[1],marker='o',capsize=5,capthick=2, c='blue',label=labels[1])

			ax.errorbar([(max_dur+min_dur)/2],[mean_opp[2]],yerr=std_opp[2],marker='o',capsize=5,capthick=2,c='green',label=labels[2])

			ax.errorbar([(max_dur+min_dur)/2],[mean_opp[3]],yerr=std_opp[3],marker='o',capsize=5,capthick=2,c='red',label=labels[3])
			#-----
			ax.errorbar([(max_dur+min_dur)/2],[mean_orac[1]],yerr=std_orac[1],marker='o',capsize=5,capthick=2, c='blue',label=labels_o[1],alpha=0.3)

			ax.errorbar([(max_dur+min_dur)/2],[mean_orac[2]],yerr=std_orac[2],marker='o',capsize=5,capthick=2,c='green',label=labels_o[2],alpha=0.3)

			ax.errorbar([(max_dur+min_dur)/2],[mean_orac[3]],yerr=std_orac[3],marker='o',capsize=5,capthick=2,c='red',label=labels_o[3],alpha=0.3)
			legend_plotted = True
	ax.set_ylim([0,1])
	ax.grid()
	ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
	ax.set_title(f"High Energy Activities + Four Low: {activities}")
	ax.set_xlabel("Average Activity Duration")
	ax.set_ylabel("Accuracy (mean and std)")
	# Create custom legend manually based on the color and marker style
	plt.tight_layout()
	plt.savefig(f"High Energy Activities: {activities} with oracle", bbox_inches='tight')
	# plt.show()

	