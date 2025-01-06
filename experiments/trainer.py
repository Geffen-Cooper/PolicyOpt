import os
import torch
import matplotlib.pyplot as plt

from datetime import datetime
from experiments.models import SimpleNet
from datasets.energy_harvest import EnergyHarvester
from datasets.apply_policy_nathan import Device

from experiments.dataloader import load_user_data

class DeviceTrainer():
	def __init__(self, exp_name, optimizer_cfg, sensor_cfg, train_cfg, classifier_cfg, device, load_path, data_path, val_user, seed):
		self.optimizer_cfg = optimizer_cfg

		self.load_path = load_path
		self.seed = seed
		self.device = device

		self.train_cfg = train_cfg

		self._setup_paths(load_path, exp_name)
		self._load_data(data_path, val_user)
		self._load_classifier(**classifier_cfg)
		self._load_sensor(**sensor_cfg)
		self._build_optimizer()

		self.fig, self.axs = plt.subplots(1,1, figsize=(20,5))

	def _setup_paths(self, load_path, exp_name):
		self.root_dir = os.path.dirname(os.path.dirname(__file__))
		if load_path is None:
			now = datetime.now()
			start_time = now.strftime("%Y-%m-%d_%H-%M-%S")
			self.log_dir = os.path.join(self.root_dir,"saved_data/runs",f"{exp_name}")+"_"+start_time
		else:
			self.log_dir = load_path
		
		# path where model parameters will be saved
		self.sensor_path = os.path.join(self.log_dir, "model_params.pt")
		# self.data_dir = os.path.join(self.root_dir,"datasets/dsads_contig/merged_preprocess")
		self.data_dir = os.path.join(self.root_dir, "saved_data/processed_data/LOOCV_preprocessed_data")

		self.plot_dir = os.path.join(self.log_dir, "plots")

		if not os.path.isdir(self.plot_dir): 
			os.makedirs(self.plot_dir)
	
	def _load_data(self, data_dir, val_user=0):   
		# self.data = load_merged_data(self.data_dir, self.device)
		self.data = load_user_data(data_dir, val_user, self.device)
		train_data = self.data['train'][0]
		# Compute mean and std used to normalize sensor data fed to classifier
		self.mean = torch.mean(train_data, dim=0)
		self.std = torch.std(train_data, dim=0)
	
	def _build_optimizer(self):
		self.opt = torch.optim.Adam(self.sensor.parameters(),lr=self.optimizer_cfg['lr'])

	def _load_classifier(self, path, num_activities):
		self.classifier = SimpleNet(3,num_activities).to(self.device)
		ckpt_path = os.path.join(self.root_dir,path)
		self.classifier.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

	def _load_sensor(self, packet_size, leakage, init_overhead, duration_range, history_size, sample_frequency, sensor_net_cfg):
		self.eh = EnergyHarvester()
		self.sensor = Device(
			packet_size=packet_size,
			leakage=leakage,
			init_overhead=init_overhead,
			eh=self.eh,
			policy_mode=self.optimizer_cfg['policy_mode'],
			classifier=self.classifier,
			device=self.device, 
			duration_range=duration_range,
			history_size=history_size, 
			sample_frequency=sample_frequency,
			mean=self.mean,
			std=self.std,
			sensor_net_cfg=sensor_net_cfg,
			seed=self.seed,
		)	
	
	def load_policy(self):
		self.sensor = self.sensor.load_state_dict(torch.load(self.sensor_path)['model_state_dict'])