import pandas as pd
import numpy as np
from enum import Enum
import torch
from datasets.energy_harvest import EnergyHarvester

class DeviceState(Enum):
	OFF = 0 # init or device energy went to 0
	ON_CAN_TX = 1 # already on and have at least thresh energy
	ON_CANT_TX = 2 # already on but have less than thresh energy
	ON_CANT_TX_DELAY = 3 # have enough energy to send but delaying some time


def sparsify_data(data_window: np.ndarray, packet_size: int, leakage: float, init_overhead: float, \
				  eh: EnergyHarvester, policy='opportunistic', learned_policy=None, train_mode=True, past_policy_outputs=[], past_actions=[], device="auto", history_size=1):
	""" Converts a 3 axis har signal into a sparse version based on energy harvested.

	In general: e_t+1 = e_t + e_h - e_l
				when the device transmits a packet, then -e_tx will be applied as well

	Parameters
	----------

	data_window: np.ndarray
		A 3 x T data array from one body part with a 3-axis accelerometer data.
		The extra column (at column 0) is the time in seconds for each sample. 
		Example columns: time | armX | armY | armZ

	packet_size: int
		number of samples in a packet

	leakage: float
		Average power consumption when device is idle (sleep mode). Each time step dt,
		leakage*dt J of energy is used since the device is almost always idle unless it
		is doing sampling and TX of a packet

	init_overhead: float
		The amount of energy required to cold start the device and initialize it

	eh: EnergyHarvester
		An energy harvester object used to derive the energy from the accelerometer trace

	policy: str
		The energy spending policy to use to decide when to transmit packets

	learned_policy: nn.Module
		A PyTorch model which takes energy traces as input and outputs a binary decision
		of whether to send a packet (0 -> don't sent, 1 -> send)

	train_mode: bool
		if we are training, we need to keep track of policy outputs so we can backprop

	Returns
	-------

	sparse_data: tuple
		TODO
	"""
	
	# applied each time step
	LEAKAGE_PER_SAMPLE = leakage*(data_window[1,0]-data_window[0,0]) # leakage_power * 1/fs --> 1/fs is dt

	# store the sampled packets
	packets = None

	# keep track of time delayed when don't send a packet
	delay_k = 0

	# how much to delay if don't send
	delay_size = 25

	# create pandas data frame as specified by EnergyHarvester.power() function
	channels = np.array([0,1,2,3]) # time + 3 acc channels of body part
	df = pd.DataFrame(data_window[:,channels],columns=['time', 'x', 'y','z'])
	
	# get energy as function of samples
	t_out, p_out = eh.power(df)
	e_out = eh.energy(t_out, p_out)
	valid, thresh = eh.generate_valid_mask(e_out, packet_size)

	# assume max energy we can store is ~ 3*thresh needed to sample/TX
	MAX_E = 4*thresh

	# create a mask of seen and unseen data
	valid = np.empty(len(e_out))
	valid[:] = np.nan

	# energy harvested each time step
	e_harvest = np.concatenate([np.array([0]),np.diff(e_out)])

	# energy state at each time step
	e_trace = np.zeros(len(e_out))

	# history of energy values, used as input to policy
	# 0-history_size -> energy values for last packet
	# history_size -> time steps since last packet
	# history_size+1-2*history_size+1 -> energy values for current packet
	# note that if the device dies, we apply opportunistic to get the first packet
		# then use the policy for all subsequent packets
	# initialize with -1
	e_input = np.zeros(2*history_size+1)-1
	last_packet_k = -1

	# device state
	STATE = DeviceState.OFF

	# assume a linear energy usage over the course of a packet
	# i.e., the quantity thresh/packet_size id used per sample. 
	linear_usage = np.linspace(0,thresh,packet_size+1)[1:]

	# keep track of model outputs so can apply loss and backprop
	policy_outputs = past_policy_outputs
	actions = past_actions

	# energy starts from 0 at time step 0 so simulate from timestep 1
	k=1
    
	# iterate over energy values
	while k < len(e_trace):
		APPLIED_POLICY = 0
		# update energy state
		e_trace[k] = e_trace[k-1] + e_harvest[k] - LEAKAGE_PER_SAMPLE
		# print(k, e_trace[k],STATE,thresh,e_harvest[k],LEAKAGE_PER_SAMPLE)

		# saturate if exceed max or becomes negative
		if e_trace[k] > MAX_E:
			e_trace[k] = MAX_E
		elif e_trace[k] < 0:
			e_trace[k] = 0

		
		''' ---------- Opportunistic Policy'''
		if policy == 'opportunistic':
			# update device state

			# OFF -> ON_CANT_TX
			if STATE == DeviceState.OFF: # turn on when have init overhead
				# OFF -> ON_CANT_TX
				if e_trace[k] >= 5*LEAKAGE_PER_SAMPLE + init_overhead:
					STATE = DeviceState.ON_CANT_TX
					try:
						e_trace[k+1] = e_trace[k] - init_overhead # apply overhead instantly
					except:
						break
					k += 2
				# OFF -> OFF
				else:
					k += 1
			# ON_CANT_TX -> ON_CAN_TX, ON_CANT_TX -> OFF, ON_CANT_TX -> ON_CANT_TX
			elif STATE == DeviceState.ON_CANT_TX:
				# ON_CANT_TX -> ON_CAN_TX
				if e_trace[k] >= thresh + 5*LEAKAGE_PER_SAMPLE:
					STATE = DeviceState.ON_CAN_TX
					# we are within one packet of the end of the data
					if k + packet_size + 1 >= len(e_trace):
						valid[k+1:] = 1
						e_trace[k+1:] = (-linear_usage[:len(e_trace)-k-1] + e_harvest[k+1:]) + e_trace[k]
						k += (packet_size+1)
						break
					# once thresh is reached, we start sampling on the next sample
					valid[k+1:k+1+packet_size] = 1

					# we apply linear energy usage for each sample and get harvested amount each step
					e_trace[k+1:k+1+packet_size] = (-linear_usage[:] + e_harvest[k+1:k+1+packet_size]) + e_trace[k]
					
					k += (packet_size+1)
				# ON_CANT_TX -> OFF
				elif e_trace[k] == 0:
					STATE = DeviceState.OFF
					k += 1
				# ON_CANT_TX -> ON_CANT_TX
				else:
					STATE = DeviceState.ON_CANT_TX
					k += 1
			# ON_CAN_TX -> OFF, ON_CAN_TX -> ON_CANT_TX
			elif STATE == DeviceState.ON_CAN_TX:
				if e_trace[k] == 0: # device died
					STATE = DeviceState.OFF
					k += 1
				elif e_trace[k] < thresh:
					STATE = DeviceState.ON_CANT_TX
					k += 1

		''' ---------- Learned Policy'''
		if policy == 'learned_policy':
			# update device state

			if STATE == DeviceState.OFF: # turn on when have init overhead
				# OFF -> ON_CANT_TX
				if e_trace[k] >= 5*LEAKAGE_PER_SAMPLE + init_overhead:
					STATE = DeviceState.ON_CANT_TX
					try:
						e_trace[k+1] = e_trace[k] - init_overhead # apply overhead instantly
					except:
						break
					k += 2
				# OFF -> OFF
				else:
					k += 1
			# ON_CANT_TX -> ON_CAN_TX, ON_CANT_TX -> OFF, ON_CANT_TX -> ON_CANT_TX
			elif STATE == DeviceState.ON_CANT_TX or (STATE == DeviceState.ON_CANT_TX_DELAY and (k - delay_k) >= delay_size):
				'''modification from opportunistic, check policy if to send, otherwise delay by a packet'''
				# ON_CANT_TX -> ON_CAN_TX
				if (e_trace[k] >= thresh + 5*LEAKAGE_PER_SAMPLE):
					# if we haven't observed a packet yet, just opportuistically sample the first one
					if e_input[0] == -1:
						STATE = DeviceState.ON_CAN_TX
						# we are within one packet of the end of the data
						if k + packet_size + 1 >= len(e_trace):
							valid[k+1:] = 1
							e_trace[k+1:] = (-linear_usage[:len(e_trace)-k-1] + e_harvest[k+1:]) + e_trace[k]
							k += (packet_size+1)
							break
						# once thresh is reached, we start sampling on the next sample
						valid[k+1:k+1+packet_size] = 1

						# we apply linear energy usage for each sample and get harvested amount each step
						e_trace[k+1:k+1+packet_size] = (-linear_usage[:] + e_harvest[k+1:k+1+packet_size]) + e_trace[k]
						
						k += (packet_size+1)

						# store last history_size energy values
						e_input[:history_size] = e_trace[k-history_size:k]
						last_packet_k = k
					# otherwise we can apply the policy
					else: 
						# store last history_size energy values for new packet 
						# TODO: this is where the input is
						e_input[history_size+1:] = e_trace[k-history_size:k]
						e_input[history_size] = (k - last_packet_k)/k # divide by k to make between [0,1]
						e_input = (e_input-np.mean(e_input))/(np.std(e_input)+1e-5)
						# if decide not to send, then delay, otherwise send
						# rand_dec = np.random.uniform()
						if train_mode:
							policy_inputs = torch.cat([
								torch.tensor(e_input[-history_size:], dtype=torch.float32, requires_grad=True, device=device), 
								torch.tensor(policy_outputs[-history_size:], dtype=torch.float32, requires_grad=True, device=device)
                            ])
							policy_out = learned_policy(policy_inputs)
							policy_outputs = torch.cat([policy_outputs, policy_out])
							APPLIED_POLICY = 1
						# else:
						# 	# Else we did not send so output is 0
						# 	# with torch.no_grad():
						# 	# 	policy_out = learned_policy(torch.tensor(e_input).float())
						# 	policy_out = torch.tensor(0.0, dtype=torch.float32)
						# 	policy_outputs.append(policy_out)

						# 0 is delay, 1 is send
						# if random is less than policy out, then we picked 1, otherwise 0 # TODO: why??? 
						# What is policy out?
						actions = torch.cat([actions, policy_out > 0.5])
						if policy_out < 0.5:
							STATE = DeviceState.ON_CANT_TX_DELAY
							delay_k = k
							k += 1
							continue
						# since policy_opt > 0.5, send packet:
						STATE = DeviceState.ON_CAN_TX
						# we are within one packet of the end of the data
						if k + packet_size + 1 >= len(e_trace):
							valid[k+1:] = 1
							e_trace[k+1:] = (-linear_usage[:len(e_trace)-k-1] + e_harvest[k+1:]) + e_trace[k]
							k += (packet_size+1)
							break
						# once thresh is reached, we start sampling on the next sample
						valid[k+1:k+1+packet_size] = 1

						# we apply linear energy usage for each sample and get harvested amount each step
						e_trace[k+1:k+1+packet_size] = (-linear_usage[:] + e_harvest[k+1:k+1+packet_size]) + e_trace[k]
						
						k += (packet_size+1)

						# store last history_size energy values
						e_input[:history_size] = e_trace[k-history_size:k]
						last_packet_k = k
				# ON_CANT_TX -> OFF
				elif e_trace[k] == 0:
					STATE = DeviceState.OFF
					k += 1
				# ON_CANT_TX -> ON_CANT_TX
				else:
					STATE = DeviceState.ON_CANT_TX
					k += 1
			# Merged with DeviceState.ON_CANT_TX		
			# elif STATE == DeviceState.ON_CANT_TX_DELAY: # stay delayed for some amount of time
			# 	# print(bp,k,STATE)
			# 	if (k - delay_k) >= delay_size:
			# 		STATE = DeviceState.ON_CAN_TX
			# 		# need to check policy again
			# 		k += 1
			# 	else:
			# 		k += 1
			
			# ON_CAN_TX -> OFF, ON_CAN_TX -> ON_CANT_TX
			elif STATE == DeviceState.ON_CAN_TX:
				if e_trace[k] == 0: # device died
					STATE = DeviceState.OFF
					k += 1
				elif e_trace[k] < thresh:
					STATE = DeviceState.ON_CANT_TX
					k += 1
				else:
					STATE = DeviceState.ON_CANT_TX
					k += 1
					
			if not APPLIED_POLICY:
				policy_out = torch.tensor([0.0], dtype=torch.float32, device=device)
				policy_outputs = torch.cat([policy_outputs, policy_out])
				actions = torch.cat([actions, policy_out])

		''' ----------- Package Data after applying policies -------- '''		
		# masking the data based on energy
		for acc in 'xyz':
			df[acc+'_eh'] = df[acc] * valid

		# get the transition points of the masked data to see where packets start and end
		og_data = df[acc+'_eh'].values
		rolled_data = np.roll(og_data, 1)
		rolled_data[0] = np.nan # in case we end halfway through a valid packet
		nan_to_num_transition_indices = np.where(~np.isnan(og_data) & np.isnan(rolled_data))[0] # arrival idxs
		num_to_nan_transition_indices = np.where(np.isnan(og_data) & ~np.isnan(rolled_data))[0] # ending idxs
		
		# now get the actually sampled data as a list of windows
		arr = df[['x_eh','y_eh','z_eh']].values
		packet_data = [                                                                               # this zip operation is important because if we end halfway through a packet it is skipped (number of starts and ends must match)
						arr[packet_start_idx : packet_end_idx] for packet_start_idx,packet_end_idx in zip(nan_to_num_transition_indices,num_to_nan_transition_indices)
						]
		
		# get the arrival time of each packet (note that the arrival time is the end of the data)
		time_idxs = df['time']
		arrival_times = [ 
						time_idxs[packet_end_idx-1] for packet_end_idx in num_to_nan_transition_indices
						]
		
		# each item in the list is a packet_size x 3 array, so we just stack into one array
		if len(packet_data) > 0:
			packet_data = np.stack(packet_data)
			
		# we make the list into an array of packet_size x 1
		if len(arrival_times) > 0:
			arrival_times = np.stack(arrival_times)
		else:
			arrival_times = np.array(arrival_times)
		
		# store as a tuple
		# entry 0 is P x 1 and entry 1 is P x packet_size x 3
		packets = (arrival_times,packet_data)
	# import matplotlib.pyplot as plt
	# plt.plot(e_out)
	# plt.plot(e_trace)
	# plt.axhline(thresh)
	# plt.axhline(init_overhead)
	# plt.show()

	if train_mode:
		return packets, e_trace, policy_outputs, actions
	else:
		return packets, e_trace


# given the output of a policy, get the classification results
def classify_packets(raw_data, labels, packets, classifier, window_size):
	
	first_sample_idx = int(packets[0][0]*25)

	# classify every single sample
	num_windows = len(labels) - first_sample_idx
	dense_outputs = []
	dense_preds = torch.zeros(num_windows)
	dense_targets = torch.zeros(num_windows)
	for win_i in range(num_windows):
		sample_idx = win_i+first_sample_idx
		# print(first_sample_idx,win_i,num_windows,sample_idx)
		win = torch.tensor(raw_data[sample_idx-window_size+1:sample_idx+1,:]).float().T.unsqueeze(0)
		dense_targets[win_i] = labels[sample_idx-window_size+1] # last sample in packet
		# make prediction
		with torch.no_grad():
			out = classifier(win)
		dense_outputs.append(out)
		dense_preds[win_i] = torch.argmax(out)
	dense_outputs = torch.stack(dense_outputs)
		
	# classify provided packet and extend predictions
	dense_outputs_policy = []
	dense_preds_policy = torch.zeros(num_windows)
	last_prediction_idx = 0
	last_pred = None
	last_out = None
	count = 0
	for packet_i, (at,win) in enumerate(zip(packets[0],packets[1])):
		# get next window
		sample_idx = int(at*25)
		
		win = torch.tensor(win).float().T.unsqueeze(0)
		
		# extend previous prediction
		if packet_i > 0:
			count += sample_idx-last_prediction_idx
			# extend output to duration of prediction
			dense_outputs_policy.append(last_out.repeat(sample_idx-last_prediction_idx,1))
			dense_preds_policy[last_prediction_idx:sample_idx] = last_pred

		# print(f"first_sample_idx:{first_sample_idx}, packet_i:{packet_i}, at_sample:{int(at*25)}, sample_idx:{sample_idx}, count: {count}")

		# make prediction
		with torch.no_grad():
			out = classifier(win)
		last_out = out
		last_pred = torch.argmax(out)

		last_prediction_idx = sample_idx

	# extend on the last one
	count += (len(labels)-last_prediction_idx)
	dense_outputs_policy.append(last_out.repeat(len(labels)-last_prediction_idx,1))
	dense_preds_policy[last_prediction_idx:] = last_pred

	# print(f"first_sample_idx:{first_sample_idx}, packet_i:{packet_i}, at_sample:{int(at*25)}, sample_idx:{sample_idx}, count: {count}")


	dense_outputs_policy = torch.cat(dense_outputs_policy)
	# print(dense_outputs_policy.shape,dense_preds_policy.shape,labels[first_sample_idx:].shape)

	dense_targets_policy = labels[first_sample_idx:]

	# from these we can get the loss over the sequence, the accuracy, etc
	# we return the whole thing because we may want to visualize what predictions would have been for unseen data
	return dense_outputs, dense_preds, dense_targets, dense_outputs_policy, dense_preds_policy, dense_targets_policy
		

	