import pandas as pd
import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.energy_harvest import EnergyHarvester
from experiments.models import DiscreteQNetwork

class DeviceState(Enum):
	OFF = 0 # init or device energy went to 0
	ON_CAN_TX = 1 # already on and have at least thresh energy
	ON_CANT_TX = 2 # already on but have less than thresh energy
	ON_CANT_TX_DELAY = 3 # have enough energy to send but delaying some time

class Device(nn.Module):
	def __init__(self, 
				 packet_size, leakage, init_overhead, eh, policy_mode, duration_range,
				 history_size, sample_frequency,
				 sensor_net_cfg, classifier,
				 device, seed):
		super().__init__()
		self.packet_size = packet_size
		self.init_overhead = init_overhead
		self.leakage = leakage
		self.eh = eh
		self.policy_mode = policy_mode
		self.device = device

		self.DURATION_RANGE = duration_range
		self.HISTORY_SIZE = history_size
		self.FS = sample_frequency

		self.classifier = classifier
		self._build_nets(sensor_net_cfg)
		
		self.g = torch.Generator()
		self.g.manual_seed(seed)
	
	def _build_nets(self, sensor_net_cfg):
		if self.policy_mode == "conservative":
			""" Learned threshold energy """
			self.alpha = nn.Parameter(torch.Tensor([1e-5]), requires_grad=True)
		elif self.policy_mode == "opportunistic":
			self.alpha = torch.Tensor([0.0])
		elif self.policy_mode == "learned_policy":
			self.alpha = torch.Tensor([0.0])
			self.Q_network = DiscreteQNetwork(2, **sensor_net_cfg)
	
	def preprocess_data(self, data_window):
		# create pandas data frame as specified by EnergyHarvester.power() function
		channels = np.array([0,1,2,3]) # time + 3 acc channels of body part
		df = pd.DataFrame(data_window[:,channels],columns=['time', 'x', 'y','z'])
		return df
	
	def preprocess_constants(self, df):
		self.dt = df['time'][1] - df['time'][0]
		self.LEAKAGE_PER_SAMPLE = self.leakage*self.dt
		# get energy as function of samples
		t_out, p_out = self.eh.power(df)
		self.e_out = self.eh.energy(t_out, p_out)
		valid, self.thresh = self.eh.generate_valid_mask(self.e_out, self.packet_size)
		# Max energy of sensor
		self.MAX_E = 4*self.thresh
		# Simulation time T
		self.T = len(self.e_out)

		# assume a linear energy usage over the course of a packet
		# i.e., the quantity thresh/packet_size id used per sample. 
		self.linear_usage = torch.linspace(0,self.thresh+self.alpha.item(),self.packet_size+1)[1:]
	
	def _sample_segment(self, data, labels):
		duration = torch.randint(low=self.DURATION_RANGE[0], high=self.DURATION_RANGE[1], size=(), generator=self.g, device=self.device)*self.FS
		# rand_start = int(rng.random()*len(labels))
		rand_start = torch.randint(high=len(labels), size=(), generator=self.g, device=self.device)
		# make sure segment doesn't exceed end of data
		if rand_start + duration >= len(labels):
			rand_start = len(labels) - duration

		data_seg = data[rand_start:rand_start+duration,:]
		label_seg = labels[rand_start:rand_start+duration]

		return data_seg, label_seg

	def forward_sensor(self, data, policy_mode=None, training=True):

		if policy_mode is None:
			policy_mode = self.policy_mode
		
		df = self.preprocess_data(data)
		self.preprocess_constants(df)
		# create a mask of seen and unseen data
		valid = np.empty(self.T)
		valid[:] = np.nan

		# energy harvested each time step
		e_harvest = np.concatenate([np.array([0]),np.diff(self.e_out)])

		# energy state at each time step
		e_trace = np.zeros(self.T)
		e_input = torch.zeros(self.HISTORY_SIZE)-1

		# device state
		STATE = DeviceState.OFF

		# keep track of model outputs so can apply loss and backprop
		policy_outputs = torch.zeros(self.T, dtype=torch.float32, device=self.device)
		actions = torch.zeros(self.T, dtype=torch.bool, device=self.device)

		# energy starts from 0 at time step 0 so simulate from timestep 1
		k=1

		startup_time = len(e_trace)
		
		# iterate over energy values
		while k < len(e_trace):
			# update energy state
			e_trace[k] = e_trace[k-1] + e_harvest[k] - self.LEAKAGE_PER_SAMPLE
			# print(k, e_trace[k],STATE,thresh,e_harvest[k],LEAKAGE_PER_SAMPLE)

			# saturate if exceed max or becomes negative
			if e_trace[k] > self.MAX_E:
				e_trace[k] = self.MAX_E
			elif e_trace[k] < 0:
				e_trace[k] = 0
			
			'''Opportunistic Policy'''
			if policy_mode == 'opportunistic' or policy_mode == 'conservative':
				# update device state

				# OFF -> ON_CANT_TX
				if STATE == DeviceState.OFF: # turn on when have init overhead
					# OFF -> ON_CANT_TX
					if e_trace[k] >= 5*self.LEAKAGE_PER_SAMPLE + self.init_overhead:
						STATE = DeviceState.ON_CANT_TX
						try:
							e_trace[k+1] = e_trace[k] - self.init_overhead # apply overhead instantly
						except:
							break
						startup_time = k
						k += 2
					# OFF -> OFF
					else:
						k += 1
				# ON_CANT_TX -> ON_CAN_TX, ON_CANT_TX -> OFF, ON_CANT_TX -> ON_CANT_TX
				elif STATE == DeviceState.ON_CANT_TX:
					# ON_CANT_TX -> ON_CAN_TX
					if e_trace[k] >= self.thresh + self.alpha + 5*self.LEAKAGE_PER_SAMPLE:
						STATE = DeviceState.ON_CAN_TX
						# we are within one packet of the end of the data
						if k + self.packet_size + 1 >= len(e_trace):
							# actions[k] = True
							# valid[k+1:] = 1
							# e_trace[k+1:] = (-linear_usage[:len(e_trace)-k-1] + e_harvest[k+1:]) + e_trace[k]
							k += (self.packet_size+1)
							break
						# once thresh is reached, we start sampling on the next sample
						actions[k] = True
						valid[k+1:k+1+self.packet_size] = 1

						# Set policy_outputs[k] = 1.0 since policy sent packet
						updated_policy_outputs = policy_outputs.clone()
						updated_policy_outputs[k] = 1.0
						policy_outputs = updated_policy_outputs

						# we apply linear energy usage for each sample and get harvested amount each step
						e_trace[k+1:k+1+self.packet_size] = (-self.linear_usage[:] + e_harvest[k+1:k+1+self.packet_size]) + e_trace[k]
						
						k += (self.packet_size+1)
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
					elif e_trace[k] < self.thresh + self.alpha:
						STATE = DeviceState.ON_CANT_TX
						k += 1

			'''Learned Policy'''
			if policy_mode == 'learned_policy':
				if STATE == DeviceState.OFF: # turn on when have init overhead
					# OFF -> ON_CAN_TX or OFF -> ON_CANT_TX
					if e_trace[k] >= 5*self.LEAKAGE_PER_SAMPLE + self.init_overhead:
						STATE = DeviceState.ON_CAN_TX
						try:
							e_trace[k+1] = e_trace[k] - self.init_overhead # apply overhead instantly
							e_trace[k+2] = e_trace[k+1] - self.LEAKAGE_PER_SAMPLE
						except:
							break
						startup_time = k
						k += 2
					# OFF -> OFF
					else:
						k += 1
			
				# ON_CANT_TX -> ON_CAN_TX, ON_CAN_TX -> ON_CANT_TX
				elif STATE == DeviceState.ON_CAN_TX:
					'''modification from opportunistic, check policy if to send, otherwise delay by a packet'''
					# ON_CANT_TX -> ON_CAN_TX
					if (e_trace[k] >= self.thresh + 5*self.LEAKAGE_PER_SAMPLE) and k > self.HISTORY_SIZE-1:
						if (torch.count_nonzero(actions) == 0):
							# First sample a packet opportunistically
							# we are within one packet of the end of the data
							if k + self.packet_size + 1 >= len(e_trace):
								actions[k] = True
								valid[k+1:] = 1
								e_trace[k+1:] = (-self.linear_usage[:len(e_trace)-k-1] + e_harvest[k+1:]) + e_trace[k]
								k += (self.packet_size+1)
								break
							# once thresh is reached, we start sampling on the next sample
							actions[k] = True
							valid[k+1:k+1+self.packet_size] = 1

							# we apply linear energy usage for each sample and get harvested amount each step
							e_trace[k+1:k+1+self.packet_size] = (-self.linear_usage[:] + e_harvest[k+1:k+1+self.packet_size]) + e_trace[k]

							updated_policy_outputs = policy_outputs.clone()
							updated_policy_outputs[k] = 1.0
							policy_outputs = updated_policy_outputs
							
							k += (self.packet_size+1)

							continue
						
						# else run the policy
						e_input = e_trace[k-self.HISTORY_SIZE+1:k+1]
						# e_input = (e_input-np.mean(e_input))/(np.std(e_input)+1e-5) # TODO: can normalize later

						# policy inputs are buffers of the last HISTORY_SIZE energy traces and policy outputs 
						policy_inputs = torch.cat([
							torch.tensor(e_input, dtype=torch.float32, device=self.device), 
							policy_outputs[k-self.HISTORY_SIZE:k]
						])

						action_values = self.Q_network(policy_inputs)
						if torch.argmax(action_values) == 0:
							# The policy chose not to sample
							k += 1
							continue
						else:
							# The policy chose to sample
							if k + self.packet_size + 1 >= len(e_trace):
								# since we are within one packet of the end of the data, we do not sample
								k += (self.packet_size+1)
								break
							else:
								actions[k] = True
								valid[k+1:k+1+self.packet_size] = 1

								# we apply linear energy usage for each sample and get harvested amount each step
								e_trace[k+1:k+1+self.packet_size] = (-self.linear_usage[:] + e_harvest[k+1:k+1+self.packet_size]) + e_trace[k]
								
								k += (self.packet_size+1)

					# ON_CAN_TX -> OFF, ON_CANT_TX -> OFF, ON
					elif e_trace[k] == 0:
						STATE = DeviceState.OFF 
						k += 1
					else:
						k += 1
				else:
					k += 1
		
		packets = self._obtain_packets(df, valid, actions)

		if training:
			return packets, e_trace, actions 
		else:
			return packets, e_trace
		
	def _obtain_packets(self, df, valid, actions):
		''' ----------- Package Data after applying policies -------- '''		
		# masking the data based on energy
		for acc in 'xyz':
			df[acc+'_eh'] = df[acc] * valid

		# get the transition points of the masked data to see where packets start and end
		og_data = df[acc+'_eh'].values
		# This is the delay. After policy sends the actual data starts sending one timestep afterwards
		rolled_data = np.roll(og_data, 1) # in case we end halfway through a valid packet.
		rolled_data[0] = np.nan
		nan_to_num_transition_indices = np.where(~np.isnan(og_data) & np.isnan(rolled_data))[0] # arrival idxs
		num_to_nan_transition_indices = np.where(np.isnan(og_data) & ~np.isnan(rolled_data))[0] # ending idxs
		
		# now get the actually sampled data as a list of windows
		arr = torch.tensor(df[['x_eh','y_eh','z_eh']].values, dtype=torch.float32, device=self.device)
		packet_data = [                                                                               
			# this zip operation is important because if we end halfway through a packet it is skipped (number of starts and ends must match)
			arr[packet_start_idx : packet_end_idx] for packet_start_idx,packet_end_idx in zip(nan_to_num_transition_indices,num_to_nan_transition_indices)
		]
		
		# get the arrival time of each packet (note that the arrival time is the end of the data)
		# time_idxs = torch.arange(len(df['time'].values), dtype=torch.float32, device=self.device, requires_grad=True)
		
		# Get packet start and end indices
		packet_start_idx = torch.roll(actions,1)
		packet_start_idx[0] = 0
		packet_end_idx = torch.roll(actions,1+self.packet_size)

		# Make the data mask 
		# Fill in the entries between [packet_start_idx, packet_end_idx] of a bool tensor with 1s
		data_mask = packet_start_idx.cumsum(dim=0) * packet_end_idx.flip(dims=(0,)).cumsum(dim=0).flip(dims=(0,))
		data_mask = data_mask.type(torch.bool).reshape(-1,1)
		data_mask = data_mask.expand(-1,arr.shape[1])
		
		if len(packet_data) != 0:
			sent_times = torch.arange(len(df['time'].values))[packet_start_idx]
			packet_data = torch.stack(packet_data)
		else:
			sent_times = None
			packet_data = None

		packets = (sent_times,packet_data)

		return packets

	def forward_classifier(self, labels, packets):
		""" Given the output of a policy, get the classification results

		Parameters
		----------
		labels: torch.tensor
			(T,) tensor of window HAR labels for raw_data
		packets: (torch.tensor, torch.tensor)
			Tuple where entry 0 is the arrival times (P x 1) and entry 1 is the packet data (P x packet_size x 3)
		classifier: nn.Module
			A PyTorch model which takes sensor data as input and outputs a HAR classification decision.

		Returns
		-------
		outputs_policy: torch.tensor
		preds_policy: torch.tensor
		targets_policy: torch.tensor
		"""
		# torch.autograd.detect_anomaly(True)

		DELAY_SIZE = 1
		first_sample_idx = int(packets[0][0]*DELAY_SIZE)

		# classify every single sample
		num_windows = len(labels) - first_sample_idx
		outputs = []
		preds = []
		targets = []
		for t, data in zip(*packets):
			data = data.T.unsqueeze(0)
			# make prediction
			out = self.classifier(data) # removed torch.no_grad()
			outputs.append( torch.softmax(out, dim=1))
			preds.append(torch.argmax(torch.softmax(out, dim=1)).unsqueeze(0))
			# save target
			target = torch.tensor([labels[t-self.packet_size+1]], dtype=torch.long, device=self.device)
			targets.append(target) # last sample in packet

		# classify provided packet and extend predictions
		outputs_policy = []
		preds_policy = []
		targets_policy = []

		for i, (t,win) in enumerate(zip(*packets)):
			# get next window
			# sample_idx = int(at*DELAY_SIZE)
			if i < len(packets[0])-1:
				count = int(packets[0][i+1] - packets[0][i])
				outputs_policy.append(outputs[i].repeat(count, 1))
				preds_policy.append(preds[i].repeat(count))
			else:
				count = int(len(labels) - (packets[0][i]-self.packet_size+1))
				outputs_policy.append(outputs[i].repeat(count, 1))
				preds_policy.append(preds[i].repeat(count))

		targets_policy = labels[int(packets[0][0])-self.packet_size+1:]

		# If did not sample at all, make targets_policy = all labels and outputs_policy to be all zeros so policy incurs high loss
		if len(packets[0]) == 0:
			targets_policy = labels
			outputs_policy = torch.zeros((targets_policy.shape), dtype=torch.float32, device=self.device)
		
		return outputs_policy, preds_policy, targets_policy
	
	def forward(self, data, labels, training):
		if training:
			train_segment_data, train_segment_labels = self._sample_segment(data, labels)

			# add time axis
			train_t_axis = torch.arange(len(train_segment_labels), dtype=torch.float64, device=self.device)/self.FS
			train_t_axis = train_t_axis.reshape(-1,1)

			# add the time axis to the data
			train_full_data_window = torch.cat((train_t_axis, train_segment_data), dim=1)
			learned_packets, e_trace, actions = self.forward_sensor(train_full_data_window, training=training)

			# If nothing is sampled, return None
			if learned_packets[0] is None:
				# Policy did not sample at all
				print(f"Did not sample")
				return (None, None)
			elif learned_packets[0].shape[0] != learned_packets[1].shape[0]:
				print(f"Data error. arrive_times.shape[0] = {learned_packets[0].shape[0]} but packets.shape[0] = {learned_packets[1].shape[0]}")
				return (None, None)

			outputs_policy, classifier_preds, classifier_targets = self.forward_classifier(train_segment_labels,learned_packets)

			return classifier_preds, classifier_targets