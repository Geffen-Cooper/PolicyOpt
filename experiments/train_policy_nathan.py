import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from collections import deque, namedtuple
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from experiments.trainer import DeviceTrainer

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) 
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class RLDeviceTrainer(DeviceTrainer):
	def __init__(self, exp_name, policy_mode, sensor_cfg, train_cfg, classifier_cfg, device, load_path, lr, seed):
		super().__init__(exp_name, policy_mode, sensor_cfg, train_cfg, classifier_cfg, device, load_path, lr, seed)
		self.memory = ReplayMemory(self.train_cfg['replay_buffer_capacity'])
		self.criterion = nn.SmoothL1Loss() # the loss function

	def optimize_model(self):
		if len(self.memory) < self.train_cfg['batch_size']:
			return
		
		transitions = self.memory.sample(self.train_cfg['batch_size'])
		batch = Transition(*zip(*transitions))
		state_batch = torch.cat(batch.state)
		state_energy_batch = state_batch[:,-1,0]
		next_state_energy_batch = torch.cat(batch.next_state)[:,-1,0]
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)

		# print("State batch print", state_batch.shape)
		# print("Last energy batch shape", state_energy_batch.shape)
		# print("Action batch shape", action_batch.shape)
		# print("Reward batch shape", reward_batch.shape)

		state_energy_mask = torch.where(state_energy_batch >= self.sensor.thresh, 1, 0)
		next_state_energy_mask = torch.where(next_state_energy_batch >= self.sensor.thresh, 1, 0)

		non_final_mask = torch.tensor([], dtype=torch.bool, device=self.sensor.device)
		for b in batch.next_state:
			for s in b:
				if s.all() == 9999: # This is for None
					non_final_mask = torch.cat((non_final_mask, torch.tensor([0])))
				else:
					non_final_mask = torch.cat((non_final_mask, torch.tensor([1])))
		
		non_final_next_states = torch.cat([s for s in batch.next_state if s.all() != 9999])

		# non final next states with enough energy
		# print("non final next states shape", non_final_next_states.shape)
		# print("state energy mask shape", state_energy_mask.shape)
		# print("non final mask shape", non_final_mask.shape)
		# print("next state energy mask shape", next_state_energy_mask.shape)

		valid_mask = torch.logical_and(non_final_mask, next_state_energy_mask)
		valid_states = state_batch[state_energy_mask]
		valid_next_states = non_final_next_states[valid_mask]

		if valid_next_states.shape[0] == 0:
			print(valid_mask.all() == 0.0)
			print(valid_next_states.shape)
			return

		# print("valid states shape", valid_states.shape)
		# print("valid next states shape", valid_next_states.shape)

		action_idx = torch.where(action_batch, 1, 0).unsqueeze(1)

		print("How many times did the policy send?", sum(action_idx))
		
		state_action_values = self.sensor.Q_network(valid_states.flatten(start_dim=1)).gather(0, action_idx)
		# print("State action values shape", state_action_values.shape)
		# print("Action idx shape", action_idx.shape)

		# next_state_values = torch.zeros(self.train_cfg['batch_size'], device=self.sensor.device) # TODO
		next_state_values = torch.zeros(state_batch.shape[0], device=self.sensor.device)
		with torch.no_grad():
			# only evaluate the Q network when enough_energy_mask is valid
			valid_next_state_values = self.sensor.Q_network(valid_next_states.flatten(start_dim=1))
			# print("valid next state action values shape", valid_next_state_values.shape)
			next_state_values[valid_mask] = torch.max(valid_next_state_values, 1).values
		# Compute the expected Q values
		# print("next state values shape", next_state_values.shape)
		expected_state_action_values = (next_state_values * self.train_cfg['gamma']) + reward_batch
		# print("expected_state action values", expected_state_action_values)

		# Compute Huber loss
		loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

		return loss

	def train_one_epoch(self, iteration, writer):
		self.sensor.train()
		loss = 0.0
		train_data, train_labels = self.data['train']

		state, action, reward, next_state, truncated = self.sensor.forward(train_data, train_labels, training=True)
		if truncated: 
			return
		self.memory.push(state, action, reward, next_state)

		for batch_idx in range(self.train_cfg['batch_size']):
			batch_loss = self.optimize_model()
			if batch_loss is not None:
				loss += batch_loss
		
		if loss == 0.0:
			return
		
		# Optimize the model
		self.opt.zero_grad()
		loss.backward()
		# In-place gradient clipping
		torch.nn.utils.clip_grad_value_(self.sensor.Q_network.parameters(), 100)
		self.opt.step()
		
		# train_f1 = f1_score(
		# 	classifier_targets.detach().cpu().numpy(), classifier_preds.detach().cpu().numpy(), average='macro'
		# 	)
		
		print("Iteration: {}, loss: {:.3f}".format(iteration, loss))

		writer.add_scalar("train_metric/batch_loss", loss, iteration)
		# writer.add_scalar("train_metric/f1", train_f1, iteration)

		return loss

	def train(self):
		writer = SummaryWriter(self.log_dir)
		best_val_f1 = 0.0

		for iteration in tqdm(range(self.train_cfg['epochs'])):
			self.train_one_epoch(iteration, writer)
			# validation
			if iteration != 0 and iteration % self.train_cfg['val_every_epochs'] == 0:
				val_loss = self.validate(iteration, writer, *self.data['val'], self.train_cfg['val_iters'])
				if val_loss['f1'] > best_val_f1:
					best_val_f1 = val_loss['f1']
					torch.save({
						'epoch': iteration + 1,
						'model_state_dict': self.sensor.state_dict(),
						'val_f1': val_loss['f1'],
						'val_classification_accuracy': val_loss['avg_reward'],
					}, self.sensor_path)
		
		self.validate(self, iteration+1, writer, *self.data['test'])
	
	def validate(self, iteration, writer, data, labels, val_iterations):
		self.sensor.eval()
		learned_reward = 0.0
		opp_reward = 0.0
		val_policy_f1 = 0.0
		val_opp_f1 = 0.0

		for _ in range(val_iterations):
			with torch.no_grad():
				val_segment_data, val_segment_labels = self.sensor._sample_segment(data, labels)

				val_t_axis = np.arange(len(val_segment_labels))/self.sensor.FS
				val_t_axis = np.expand_dims(val_t_axis,axis=0).T
				val_t_axis = torch.tensor(val_t_axis, device=device)
				val_full_data_window = torch.cat((val_t_axis, val_segment_data), dim=1)

				learned_packets, learned_e_trace, actions = self.sensor.forward_sensor(val_full_data_window)
				
				if learned_packets[0] is None:
					# Policy did not sample at all
					print(f"Iteration {iteration}: Policy did not sample at all during validation!")
					continue

				opp_packets, opp_e_trace, opp_actions = self.sensor.forward_sensor(val_full_data_window, policy_mode="opportunistic")
				

				outputs_learned, preds_learned, targets_learned = self.sensor.forward_classifier(val_segment_labels,learned_packets)

				outputs_opp, preds_opp, targets_opp = self.sensor.forward_classifier(val_segment_labels,opp_packets)

				learned_reward += torch.where(preds_learned == targets_learned, 1, 0).sum() / len(preds_learned)
				opp_reward += torch.where(preds_opp == targets_opp, 1, 0).sum() / len(preds_opp)

				val_policy_f1 += f1_score(
					targets_learned.detach().cpu().numpy(), preds_learned.detach().cpu().numpy(), average='macro'
				)
				val_opp_f1 += f1_score(
					targets_opp.detach().cpu().numpy(), preds_opp.detach().cpu().numpy(), average='macro'
				)
		
		learned_reward /= val_iterations
		opp_reward /= val_iterations
		val_policy_f1 /= val_iterations
		val_opp_f1 /= val_iterations

		print("Iteration: {}, val_policy_f1_score: {:.3f}, val_opp_f1_score {:.3f}".format(iteration, val_policy_f1, val_opp_f1))

		writer.add_scalar("val_metric/f1_difference", val_policy_f1 - val_opp_f1, iteration)
		writer.add_scalar("val_metric/policy_f1", val_policy_f1, iteration)
		writer.add_scalar("val_metric/opp_f1", val_opp_f1, iteration)
		writer.add_scalar("val_metric/policy_reward", learned_reward, iteration)
		writer.add_scalar("val_metric/opp_reward", opp_reward, iteration)

		val_loss = {
			'f1': val_policy_f1,
			'avg_reward': learned_reward,
		}

		if learned_packets[0] is None:
			# Policy did not sample at all
			print(f"Iteration {iteration}: Policy did not sample at all during validation so no validation plot!")
			return val_loss
		
		policy_sample_times = (learned_packets[0]).long() - self.sensor.packet_size
		opp_sample_times = (opp_packets[0]).long() - self.sensor.packet_size
		self.axs.axhline(y=self.sensor.thresh, linestyle='--', color='green') # Opportunistic policy will send at this energy
		self.axs.plot(val_t_axis, learned_e_trace)
		self.axs.plot(val_t_axis, opp_e_trace, linestyle='--')
		self.axs.scatter(val_t_axis[policy_sample_times], learned_e_trace[policy_sample_times], label='policy')
		self.axs.scatter(val_t_axis[opp_sample_times], opp_e_trace[opp_sample_times], marker='D', label='opp')
		self.axs.set_xlabel("Time")
		self.axs.set_ylabel("Energy")
		self.axs.legend()
		plt.tight_layout()
		plt.savefig(f"{self.plot_dir}/plot_{iteration}.png")
		self.axs.cla()

		return val_loss

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--load_path", type=str, default=None)
	args = parser.parse_args()

	exp_name = "TestPolicy"
	epochs = 5_000
	load_path = args.load_path
	seed = 0
	policy_model = "MLP"
	device = "cpu"
	lr = 1e-3
	policy_mode = "learned_policy"

	sensor_net_cfg = {
		'in_dim': 32, # buffer=16, 2*buffer
		'hidden_dim': 32
	}

	# sensor_cfg = (packet_size, leakage, init_overhead, duration_range, history_size, sample_frequency)
	sensor_cfg = {
		'packet_size': 8,
		'leakage': 6e-6,
		'init_overhead': 150e-6,
		'duration_range': (10,100),
		'history_size': 16,
		'sample_frequency': 25,
		'sensor_net_cfg': sensor_net_cfg,
	}

	train_cfg = {
		'batch_size': 1,
		'epochs': 5_000,
		'val_iters': 10,
		'val_every_epochs': 25,	
		'gamma': 0.99,
		'replay_buffer_capacity': 10_000,	
	}

	classifier_cfg = {
		'path': "saved_data/checkpoints/dsads_contig/seed123_activities_[ 0  1  2  3  9 11 15 17 18].pth",
		'num_activities': 9,
	}

	trainer = RLDeviceTrainer(exp_name, policy_mode, sensor_cfg, train_cfg, classifier_cfg, device, load_path, lr, seed)
	trainer.train()


	# assert (device == "cuda" or device == "cpu")
	# assert (policy_model == "MLP" or policy_model == "ResNet")

	# assert(PACKET_SIZE <= BUFFER_SIZE), f"Packet size must be smaller than buffer size. Got packet size {PACKET_SIZE} and buffer size {BUFFER_SIZE}"
	# assert(PACKET_SIZE < DURATION_RANGE[0] * FS), f"The minimum duration range must be longer than the packet size. Got packet size {PACKET_SIZE} and min duration {DURATION_RANGE[0]}"