import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.setup_funcs import *
from datasets.dsads_contig.dsads import *
from datasets.energy_harvest import EnergyHarvester
from train import *
from datasets.apply_policy_nathan import Device
from sklearn.metrics import f1_score

from experiments.dataloader import load_data

class Trainer():
	def __init__(self, exp_name, policy_mode, sensor_cfg, train_cfg, device, load_path, lr, seed):
		self.policy_mode = policy_mode

		self.load_path = load_path
		self.seed = seed
		self.device = device

		self.train_cfg = train_cfg

		self._setup_paths(load_path, exp_name)
		self._load_data()
		self._load_classifier()
		self._load_sensor(**sensor_cfg)
		self._build_optimizer(lr)

		self.memory = ReplayMemory(self.train_cfg['replay_buffer_capacity'])

		self.fig, self.axs = plt.subplots(1,1)

	def _setup_paths(self, load_path, exp_name):
		self.root_dir = os.path.dirname(os.path.dirname(__file__))
		if load_path is None:
			now = datetime.datetime.now()
			start_time = now.strftime("%Y-%m-%d_%H-%M-%S")
			self.log_dir = os.path.join(self.root_dir,"saved_data/runs",f"{exp_name}")+"_"+start_time
		else:
			self.log_dir = load_path
		
		# path where model parameters will be saved
		self.sensor_path = os.path.join(self.log_dir, "model_params.pt")
		self.data_dir = os.path.join(self.root_dir,"datasets/dsads_contig/merged_preprocess")

		self.plot_dir = os.path.join(self.log_dir, "plots")
		if not os.path.isdir(self.plot_dir): os.makedirs(self.plot_dir)
	
	def _load_data(self):
		self.data = load_data(self.data_dir, self.device)
	
	def _build_optimizer(self, lr):
		self.opt = torch.optim.Adam(self.sensor.parameters(),lr=lr)

	def _load_classifier(self):
		self.classifier = SimpleNet(3,10).to(self.device)
		ckpt_path = os.path.join(self.root_dir,f"saved_data/checkpoints/seed{123}.pth")
		self.classifier.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

	def _load_sensor(self, packet_size, leakage, init_overhead, duration_range, history_size, sample_frequency, sensor_net_cfg):
		self.eh = EnergyHarvester()
		self.sensor = Device(
			packet_size=packet_size,
			leakage=leakage,
			init_overhead=init_overhead,
			eh=self.eh,
			policy_mode=self.policy_mode,
			classifier=self.classifier,
			device=self.device, 
			duration_range=duration_range,
			history_size=history_size, 
			sample_frequency=sample_frequency,
			sensor_net_cfg=sensor_net_cfg,
			seed=self.seed,
		)		

	def calculate_loss(self, y_hat, y) -> float:
		return sum([len(torch.where(y_traj == y_hat_traj)[0]) for (y_hat_traj,y_traj) in zip(y_hat,y)])

	def optimize_model(self):
		if len(self.memory < self.train_cfg['batch_size']):
			return
		
		transitions = self.memory.sample(self.train_cfg['batch_size'])

		batch = Transition(*zip(*transitions))
		state_batch = torch.cat(batch.state)
		last_energy_batch = torch.cat(batch.last_energy)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)

		enough_energy_mask = torch.where(last_energy_batch >= self.sensor.threshold)
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
		non_final_next_states = torch.cat([s for s in batch.next_state
													if s is not None])
		# non final next states with enough energy
		valid_states = torch.cat([s for s,e in zip(batch.state, batch.last_energy) if e > self.sensor.threshold and s is not None])
		
		state_action_values = self.sensor.Q_network(valid_states).gather(1, -1)

		next_state_values = torch.zeros(self.train_cfg['batch_size'], device=device)
		with torch.no_grad():
			# TODO: add enough_energy_mask - only feedback with decisions made when enough_energy_mask is valid
			next_state_values[non_final_mask] = self.sensor.Q_network(non_final_next_states).max(1).values
		# Compute the expected Q values
		expected_state_action_values = (next_state_values * self.train_cfg['gamma']) + reward_batch

		# Compute Huber loss
		criterion = nn.SmoothL1Loss()
		loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

		# Optimize the model
		self.opt.zero_grad()
		loss.backward()
		# In-place gradient clipping
		torch.nn.utils.clip_grad_value_(self.sensor.Q_network.parameters(), 100)
		self.opt.step()

	def train_one_epoch(self, iteration, writer):
		self.sensor.train()
		loss = 0.0
		train_data, train_labels = self.data['train']

		for batch_idx in range(self.train_cfg['batch_size']):
			self.save_transition() # TODO	
			loss += self.optimize_model()		
			# classifier_preds, classifier_targets = self.sensor.forward(train_data, train_labels, training=True)
			
			# if classifier_preds is None:
			# 	print(f"No packets sent on iteration {iteration} batch index {batch_idx}")
			# 	continue
			# loss += self.calculate_loss(classifier_preds, classifier_targets)

		if loss == 0.0:
			print("Iteration {} did not sample at all".format(iteration))
			return

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
				val_loss = self.validate(iteration, writer, *self.data['val'])
				if val_loss['f1'] > best_val_f1:
					best_val_f1 = val_loss['f1']
					torch.save({
						'epoch': iteration + 1,
						'model_state_dict': self.sensor.state_dict(),
						'val_f1': val_loss['f1'],
						'val_classification_loss': val_loss['loss'],
					}, self.sensor_path)
		
		self.validate(self, iteration+1, writer, *self.data['test'])
	
	def validate(self, iteration, writer, data, labels, val_iterations):
		self.sensor.eval()
		val_loss = 0.0
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
				
				if len(learned_packets[0]) == 0:
					# Policy did not sample at all
					print(f"Iteration {iteration}: Policy did not sample at all during validation!")
					continue

				opp_packets, opp_e_trace, opp_actions = self.sensor.forward_sensor(val_full_data_window, policy_mode="opportunistic")
				

				outputs_learned, preds_learned, targets_learned = self.sensor.forward_classifier(val_segment_data,val_segment_labels,learned_packets)

				outputs_opp, preds_opp, targets_opp = self.sensor.forward_classifier(val_segment_data,val_segment_labels,opp_packets)

				val_loss += self.calculate_loss(targets_learned, preds_learned)

				val_policy_f1 += f1_score(
					targets_learned.detach().cpu().numpy(), preds_learned.detach().cpu().numpy(), average='macro'
				)
				val_opp_f1 += f1_score(
					targets_opp.detach().cpu().numpy(), preds_opp.detach().cpu().numpy(), average='macro'
				)
		
		val_loss /= val_iterations
		val_policy_f1 /= val_iterations
		val_opp_f1 /= val_iterations

		print("Iteration: {}, val_policy_f1_score: {:.3f}, val_opp_f1_score {:.3f}".format(iteration, val_policy_f1, val_opp_f1))

		writer.add_scalar("val_metric/f1_difference", val_policy_f1 - val_opp_f1, iteration)
		writer.add_scalar("val_metric/policy_f1", val_policy_f1, iteration)
		writer.add_scalar("val_metric/opp_f1", val_opp_f1, iteration)
		writer.add_scalar("val_metric/loss", val_loss, iteration)
		
		policy_sample_times = (learned_packets[0]).long() - self.sensor.packet_size + 1
		opp_sample_times = (opp_packets[0]).long() - self.sensor.packet_size + 1
		self.axs.axhline(y=5.12e-5, linestyle='--', color='green') # Opportunistic policy will send at this energy
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

		val_loss = {
			'f1': val_policy_f1,
			'loss': val_loss,
		}

		return val_loss


	def load_policy(self):
		self.sensor = self.sensor.load_state_dict(torch.load(self.sensor_path)['model_state_dict'])

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
		'batch_size': 32,
		'epochs': 5_000,
		'val_duration': 100,
		'val_every_epochs': 50,	
		'gamma': 0.99,
		'replay_buffer_capacity': 10_000,	
	}

	trainer = Trainer(exp_name, policy_mode, sensor_cfg, train_cfg, device, load_path, lr, seed)
	trainer.train()


	# assert (device == "cuda" or device == "cpu")
	# assert (policy_model == "MLP" or policy_model == "ResNet")

	# assert(PACKET_SIZE <= BUFFER_SIZE), f"Packet size must be smaller than buffer size. Got packet size {PACKET_SIZE} and buffer size {BUFFER_SIZE}"
	# assert(PACKET_SIZE < DURATION_RANGE[0] * FS), f"The minimum duration range must be longer than the packet size. Got packet size {PACKET_SIZE} and min duration {DURATION_RANGE[0]}"