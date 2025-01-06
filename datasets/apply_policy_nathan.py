import pandas as pd
import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.energy_harvest import EnergyHarvester
from experiments.models import DiscreteQNetwork

# TODO: for testing
import matplotlib.pyplot as plt

class DeviceState(Enum):
	OFF = 0 # init or device energy went to 0
	ON = 1 # otherwise

class Device(nn.Module):
	""" 
	TODO:
	1. make into different devices depending on data transmission policy (e.g. OpportunisticDevice, ConservativeDevice, RLDevice, etc) 
	2. don't use dataframe
	"""
	def __init__(self, 
				 packet_size: int, 
				 leakage: float, 
				 init_overhead: float, 
				 eh: EnergyHarvester, 
				 policy_mode: str, 
				 duration_range: tuple,
				 history_size: int, 
				 sample_frequency: float,
				 sensor_net_cfg: dict, 
				 classifier: nn.Module,
				 mean: float, 
				 std: float,
				 device: str, 
				 seed: int):
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

		self.mean = mean.to(device=self.device)
		self.std = std.to(device=self.device)

		self.classifier = classifier
		self._build_nets(sensor_net_cfg)
		
		self.g = torch.Generator(device=self.device)
		self.g.manual_seed(seed)
	
	def _build_nets(self, sensor_net_cfg):
		if self.policy_mode == "rl":
			self.Q_network = DiscreteQNetwork(2, **sensor_net_cfg).to(device=self.device)
		else:
			pass
	
	def preprocess_data(self, data_window):
		# create pandas data frame as specified by EnergyHarvester.power() function
		channels = np.array([0,1,2,3]) # time + 3 acc channels of body part
		df = pd.DataFrame(data_window.cpu()[:,channels],columns=['time', 'x', 'y','z'])
		return df
	
	def preprocess_constants(self, df):
		self.dt = df['time'][1] - df['time'][0]
		self.LEAKAGE_PER_SAMPLE = self.leakage*self.dt
		# get energy as function of samples
		t_out, p_out = self.eh.power_pd(df)
		self.e_out = self.eh.energy(t_out, p_out)
		valid, self.thresh = self.eh.generate_valid_mask(self.e_out, self.packet_size)
		# Max energy of sensor
		self.MAX_E = 4*self.thresh
		# Simulation time T
		self.T = len(self.e_out)

		# assume a linear energy usage over the course of a packet
		# i.e., the quantity thresh/packet_size id used per sample. 
		self.linear_usage = torch.linspace(0,self.thresh,self.packet_size+1, device=self.device)[1:]
	
	def _sample_segment(self, data, labels):
		duration = torch.randint(low=self.DURATION_RANGE[0], high=self.DURATION_RANGE[1], size=(), generator=self.g, device=self.device)*self.FS
		rand_start = torch.randint(high=len(labels), size=(), generator=self.g, device=self.device)
		# make sure segment doesn't exceed end of data
		if rand_start + duration >= len(labels):
			rand_start = len(labels) - duration

		data_seg = data[rand_start:rand_start+duration,:]
		label_seg = labels[rand_start:rand_start+duration]

		return data_seg, label_seg
	
	def _choose_action(self, policy_mode, inputs):
		if policy_mode == "rl":
			eps_threshold = 0.95 # TODO: decreasing epsilon
			action_values = inputs
			sample = torch.randn((), device=self.device, generator=self.g)
			if sample > eps_threshold:
				return torch.argmax(action_values)
			else:
				# return torch.randint(low=0, high=2, size=(), device=self.device, generator=self.g)
				return torch.tensor(1, dtype=torch.long, device=self.device)
		else:
			raise NotImplementedError()

	def _send_condition(self, params):
		return True

	def forward_sensor(self, data, params=[0.0, 0.0], policy_mode=None):
		if policy_mode is None:
			policy_mode = self.policy_mode

		# Initialize params
		alpha = params[0]
		tau = int(params[1])

		if policy_mode == "opportunistic":
			alpha = 0
			tau = 0

		# Preprocess data to a Pandas dataframe		
		df = self.preprocess_data(data)
		self.preprocess_constants(df)

		# create a mask of seen and unseen data
		valid = np.empty(self.T)
		valid[:] = np.nan

		# energy harvested each time step
		e_harvest = np.concatenate([np.array([0]),np.diff(self.e_out)])
		e_harvest = torch.tensor(e_harvest, device=self.device)

		# energy state at each time step
		e_trace = torch.zeros(self.T, dtype=torch.float32, device=self.device)

		# device state
		STATE = DeviceState.OFF

		# keep track of model outputs so can apply loss and backprop
		actions = torch.zeros(self.T, dtype=torch.bool, device=self.device)

		# energy starts from 0 at time step 0 so simulate from timestep 1
		k=1
		last_sent_idx=0
		
		# iterate over energy values
		while k < len(e_trace):
			# update energy state
			e_trace[k] = e_trace[k-1] + e_harvest[k] - self.LEAKAGE_PER_SAMPLE
			# print(k, e_trace[k])
			# saturate if exceed max or becomes negative
			if e_trace[k] > self.MAX_E:
				e_trace[k] = self.MAX_E
			elif e_trace[k] < 0:
				e_trace[k] = 0

			# TODO: Generalize sensor logic to a method provided as input 
			# Input: DeviceState, e_trace[k], params

			'''Opportunistic Policy'''
			if policy_mode == 'opportunistic' or policy_mode == 'conservative':
				# update device state

				# OFF -> ON_CANT_TX
				if STATE == DeviceState.OFF: # turn on when have init overhead
					# OFF -> ON_CANT_TX
					if e_trace[k] >= 5*self.LEAKAGE_PER_SAMPLE + self.init_overhead:
						STATE = DeviceState.ON
						try:
							e_trace[k+1] = e_trace[k] - self.init_overhead # apply overhead instantly
						except:
							break
						startup_time = k
						k += 2
					# OFF -> OFF
					else:
						k += 1
				elif STATE == DeviceState.ON:
					# ON -> OFF
					if e_trace[k] == 0:
						STATE = DeviceState.OFF
						k += 1
					# Send if energy is above threshold
					elif e_trace[k] >= self.thresh + alpha + 5*self.LEAKAGE_PER_SAMPLE and (k - last_sent_idx >= tau): # TODO: replace with self._send_condition(params)
						# we are within one packet of the end of the data
						if k + self.packet_size + 1 >= len(e_trace):
							k += (self.packet_size+1)
							break
						# once thresh is reached, we start sampling on the next sample
						last_sent_idx = k
						actions[k] = True
						valid[k+1:k+1+self.packet_size] = 1
						# we apply linear energy usage for each sample and get harvested amount each step
						e_trace[k+1:k+1+self.packet_size] = (-self.linear_usage[:] + e_harvest[k+1:k+1+self.packet_size]) + e_trace[k]
						k += (self.packet_size+1)					
					else:
						k += 1
			
			'''Learned Policy'''
			if policy_mode == 'rl':
				if STATE == DeviceState.OFF: # turn on when have init overhead
					# OFF -> ON_CAN_TX or OFF -> ON_CANT_TX
					if e_trace[k] >= 5*self.LEAKAGE_PER_SAMPLE + self.init_overhead:
						STATE = DeviceState.ON
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
				elif STATE == DeviceState.ON:
					'''modification from opportunistic, check policy if to send, otherwise delay by a packet'''
					# ON -> OFF
					if e_trace[k] == 0:
						STATE = DeviceState.OFF
						k += 1
					# When the sensor has enough energy to send
					elif (e_trace[k] >= self.thresh + 5*self.LEAKAGE_PER_SAMPLE) and k > self.HISTORY_SIZE-1:
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
							
							k += (self.packet_size+1)

						else:
							# else run the policy
							e_input = e_trace[k-self.HISTORY_SIZE+1:k+1]
							# e_input = (e_input-np.mean(e_input))/(np.std(e_input)+1e-5) # TODO: can normalize later

							# policy inputs are buffers of the last HISTORY_SIZE energy traces and policy outputs 
							policy_inputs = torch.cat([
								e_input, 
								actions[k-self.HISTORY_SIZE:k]
							]).to(device=self.device)

							action_values = self.Q_network(policy_inputs)
							action = self._choose_action(policy_mode, action_values)
							if action == 0:
								# The policy chose not to sample
								k += 1
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
					else:
						k += 1
				else:
					raise SystemError(f"Should not ever end up here. System state is {STATE}")
		
		packets = self._obtain_packets(df, valid, actions.cpu())

		return packets, e_trace, actions 
		
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
		arr = torch.tensor(df[['x_eh','y_eh','z_eh']].values, dtype=torch.float32).cpu()
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

			sent_times = sent_times.to(device=self.device)
			packet_data = packet_data.to(device=self.device)
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
		last_sample_idx = int(packets[0][-1])

		# classify every single sample
		num_windows = len(labels) - first_sample_idx
		outputs = []
		preds = []
		targets = []
		for t, data in zip(*packets):
			# make prediction
			with torch.no_grad():
				data = data.T.unsqueeze(0)
				data = (data-self.mean.unsqueeze(0).unsqueeze(2))/(self.std.unsqueeze(0).unsqueeze(2) + 1e-5)
				out = self.classifier(data) # removed torch.no_grad()
			outputs.append(out)
			preds.append(torch.argmax(out).unsqueeze(0))
			# save target
			target = torch.tensor([labels[t-self.packet_size+1]], dtype=torch.long, device=self.device)
			targets.append(target) # last sample in packet

		# classify provided packet and extend predictions
		outputs_policy = []
		preds_policy = torch.tensor([], dtype=torch.long, device=self.device)
		targets_policy = []

		# TODO: fix this!
		for i, (t,win) in enumerate(zip(*packets)):
			# get next window
			is_last_packet = (i == len(packets[0]) - 1)

			if not is_last_packet:
				count = int(packets[0][i+1] - packets[0][i])
				outputs_policy.append(outputs[i].repeat(count, 1))
				preds_policy = torch.cat((preds_policy, preds[i].repeat(count)))
			else:
				# If last packet, extend preds_policy to be length (len(labels))
				count = int(len(labels) - packets[0][i])
				if count > 0:
					outputs_policy.append(outputs[i].repeat(count, 1))
					preds_policy = torch.cat((preds_policy, preds[i].repeat(count)))

		targets_policy = labels[first_sample_idx : ]

		# If did not sample at all, make targets_policy = all labels and outputs_policy to be all zeros so policy incurs high loss
		if len(packets[0]) == 0:
			targets_policy = labels
			outputs_policy = torch.zeros((targets_policy.shape), dtype=torch.float32, device=self.device)
				
		return outputs_policy, preds_policy, targets_policy
	
	def forward_rl(self, data, labels, training):
		if training:
			segment_data, segment_labels = self._sample_segment(data, labels)

			# add time axis
			t_axis = torch.arange(len(segment_labels), dtype=torch.float64, device=self.device)/self.FS
			t_axis = t_axis.reshape(-1,1)

			# add the time axis to the data
			train_full_data_window = torch.cat((t_axis, segment_data), dim=1)
			learned_packets, e_trace, actions = self.forward_sensor(train_full_data_window)

			# If nothing is sampled, return None
			if learned_packets[0] is None:
				# Policy did not sample at all
				print(f"Did not sample")
				return (None, None, None, None, True)
			elif learned_packets[0].shape[0] != learned_packets[1].shape[0]:
				print(f"Data error. arrive_times.shape[0] = {learned_packets[0].shape[0]} but packets.shape[0] = {learned_packets[1].shape[0]}")
				return (None, None, None, None, True)

			outputs_policy, classifier_preds, classifier_targets = self.forward_classifier(segment_labels,learned_packets)

			# print("Classifier preds shape", classifier_preds.shape)

			rewards = torch.where(classifier_preds == classifier_targets, 1, 0)
			rewards = torch.cumsum(rewards, dim=0)	/ torch.arange(1,len(rewards)+1, device=self.device) # cumulative mean
			full_states = torch.stack((e_trace, actions), dim=1)

			states = torch.tensor([], dtype=torch.float32, device=self.device)
			for i in range(full_states.shape[0] - self.HISTORY_SIZE):
				current_state = full_states[i : i+self.HISTORY_SIZE].unsqueeze(0)
				states = torch.cat((states, current_state))
			states = torch.cat((
				states, 
	 			9999*torch.ones((1,self.HISTORY_SIZE,2), device=self.device)
				)) 

			state = states[:-1]
			actions = actions[self.HISTORY_SIZE:]
			print("rewards shape", rewards.shape, "state shape", state.shape, "full state shape", full_states.shape)
			# pad rewards with zero (it is shorter because classifier has not sent)
			rewards = torch.cat((
				torch.zeros(state.shape[0]-rewards.shape[0], device=self.device), 
				rewards
				))
			next_state = states[1:]
			truncated = False

			# print("state shape", state.shape)
			# print("actions shape", actions.shape)
			# print("next state shape", next_state.shape)
			# print("rewards shape", rewards.shape)

			return state, actions, rewards, next_state, truncated
	
	def forward_zeroth(self, params, data, labels, training):
		if training:
			learned_packets, e_trace, actions = self.forward_sensor(data, params)

			# If nothing is sampled, return None
			if learned_packets[0] is None:
				# Policy did not sample at all
				print(f"Did not sample")
				return 0.0
			elif learned_packets[0].shape[0] != learned_packets[1].shape[0]:
				print(f"Data error. arrive_times.shape[0] = {learned_packets[0].shape[0]} but packets.shape[0] = {learned_packets[1].shape[0]}")
				return 0.0

			outputs_policy, classifier_preds, classifier_targets = self.forward_classifier(labels,learned_packets)

			# print("Classifier preds shape", classifier_preds.shape)

			rewards = torch.where(classifier_preds == classifier_targets, 1, 0)
			# pad rewards with zero (it is shorter because classifier has not sent)
			rewards = torch.cat((torch.zeros(actions.shape[0]-rewards.shape[0]), rewards))
			rewards = torch.sum(rewards) / len(rewards)
			# print(rewards)

			return rewards

	def predict_activity_transitions(self, e_trace, dt, sigma=100.0, threshold=2e-6):
		"""
			Parameters are the threshold, sigma, and min number of buffer samples 
		"""
		transition_times = []

		# Perform Gaussian smoothing on the e_trace
		radius = int(sigma * 3.0)
		support = torch.arange(-radius, radius + 1, dtype=torch.float32)
		kernel = torch.distributions.Normal(loc=0, scale=sigma).log_prob(support).exp_()
		kernel *= 1 / kernel.sum()
		smoothed_e_trace = 	F.conv1d(e_trace.reshape(1,1,-1), kernel.reshape(1,1,-1)).squeeze()

		# plt.plot(e_trace.squeeze())
		# plt.plot(smoothed_e_trace)
		# plt.savefig("smoothed_e_trace.png")
		# plt.clf()

		e_dot = torch.diff(smoothed_e_trace) / dt
		e_dot_buffer = torch.tensor([])

		# plt.plot(e_dot)
		# plt.savefig("e_dot.png")
		# plt.clf()

		for i in range(len(e_dot)):
			if (e_dot_buffer.shape[0] > 10): # Min number of samples in the buffer
				if e_dot_buffer.mean()-threshold < e_dot[i] < e_dot_buffer.mean()+threshold:
					e_dot_buffer = torch.cat((e_dot_buffer, torch.tensor([e_dot[i]])))
				else:
					transition_times.append(i)
					e_dot_buffer = torch.tensor([])
			else:
				e_dot_buffer = torch.cat((e_dot_buffer, torch.tensor([e_dot[i]])))
			
		return transition_times
	
	def forward_zeroth_unlabelled(self, params, data, training, gamma:float=0.5):
		if training:
			learned_packets, e_trace, actions = self.forward_sensor(data, params)
			sent_times = learned_packets[0]
			# If nothing is sampled, return None
			if sent_times is None:
				# Policy did not sample at all
				print(f"Did not sample")
				return 0.0
			elif learned_packets[0].shape[0] != learned_packets[1].shape[0]:
				print(f"Data error. arrive_times.shape[0] = {learned_packets[0].shape[0]} but packets.shape[0] = {learned_packets[1].shape[0]}")
				return 0.0
		
			pred_activity_transitions = self.predict_activity_transitions(e_trace, self.dt)

			# plt.plot(e_trace)
			# plt.vlines(pred_activity_transitions, 0, 2e-4, colors='black', linestyles='dashed')
			# plt.vlines(sent_times, 0, 2e-4, color='red', linestyles='dotted')
			# plt.savefig("Test.png")
			# plt.clf()

			# print("Pred Num Transitions", len(pred_activity_transitions))

			rewards = 0.0
			for t0,tf in zip(pred_activity_transitions[:-1], pred_activity_transitions[1:]):
				sent_times_per_activity = 0
				for t in sent_times:
					if t0 < t < tf:
						sent_times_per_activity += 1
				discounted_sum = torch.sum(gamma ** torch.arange(sent_times_per_activity))
				# print("Sent times this interval", sent_times_per_activity)
				# print("Discounted sum", discounted_sum)
				rewards += discounted_sum
			
			rewards /= len(pred_activity_transitions) # normalize by number of predicted activity transitions
			
			return rewards