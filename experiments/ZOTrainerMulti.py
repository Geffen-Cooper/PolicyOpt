import torch
from argparse import ArgumentParser
from functools import partial
from datasets.energy_harvest import EnergyHarvester
from datasets.apply_policy_nathan import Device
from ZOTrainerSingle import ZOTrainerSingle

from experiments.dataloader import load_multisensor_data

class ZOTrainerMulti(ZOTrainerSingle):
    def __init__(self, exp_name, optimizer_cfg, sensor_cfg, train_cfg, classifier_cfg, device, load_path, data_path, val_user, seed, body_parts, mode='labeled'):
        self.body_parts = body_parts
        self.mode = mode
        super().__init__(exp_name, optimizer_cfg, sensor_cfg, train_cfg, classifier_cfg, device, load_path, data_path, val_user, seed)
        self._preprocess_data()

    def _load_data(self, data_dir, val_user=0):   
        """
            TODO:
            1. Implement multisensor data preprocessing to finish load_multisensor_data
        """
        self.data = load_multisensor_data(data_dir, val_user, self.body_parts, self.device) # TODO
        train_data = self.data['train'][0]
        # Compute mean and std used to normalize sensor data fed to classifier
        self.mean = torch.mean(train_data, dim=0)
        self.std = torch.std(train_data, dim=0)
    
    def _load_sensor(self, packet_size, leakage, init_overhead, duration_range, history_size, sample_frequency, sensor_net_cfg):
        self.eh = EnergyHarvester()
        self.sensor = {}
        for bp in self.body_parts:
            self.sensor[bp] = Device(
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

    def _build_optimizer(self):
        """
            TODO:
            1. Implement self.sensor.forward_multi_zeroth()
        """
        # Initialize optimizer
        if self.mode == 'labeled':
            f = partial(self.sensor.forward_multi_zeroth, training=True) # TODO
        elif self.mode == 'unlabeled':
            f = partial(self.sensor.forward_multi_zeroth_unlabelled, training=True) # TODO
        else:
            raise NotImplementedError(f"Mode {self.mode} is invalid")
        
        Optimizer = self.optimizer_cfg['optimizer']
        self.optimizer = Optimizer(self.optimizer_cfg['init_params'], self.optimizer_cfg['lr'], self.train_cfg['batch_size'], f, params_bounds=self.optimizer_cfg['params_bounds'])
    
    def train_one_epoch(self, iteration, writer, data, labels):
        """
            TODO
            1. Current implementation is for training all sensors together at once. Implement individual training for each sensor (this is the original single sensor code!)
        """
        [sensor.train() for sensor in self.sensor.values()]
        segment_data = {}
        segment_labels = {}
        train_full_data_window = {}
        for bp in self.body_parts:
            segment_data[bp], segment_labels[bp] = self.sensor[bp]._sample_segment(data, labels)
            # add time axis
            t_axis = torch.arange(len(segment_labels[bp]), dtype=torch.float64, device=self.device)/self.sensor.FS
            t_axis = t_axis.reshape(-1,1)
            # add the time axis to the data
            train_full_data_window[bp] = torch.cat((t_axis, segment_data[bp]), dim=1)

        if self.mode == 'labeled':
            f_args = {
                'data': train_full_data_window,
                'labels': segment_labels
            } 
        elif self.mode == 'unlabeled':
            f_args = {
                'data': train_full_data_window,
            }
        else:
            raise NotImplementedError(f"Mode {self.mode} is invalid")
        
        average_reward = self.optimize_model(f_args)
        
        print("Iteration {}: avg reward: {:.3f}, params: {}, epsilon: {}".format(iteration, average_reward, self.optimizer.params, self.optimizer.epsilon))

        writer.add_scalar("train_metric/average_reward", average_reward, iteration)
        writer.add_scalars("train_metric/params", 
                          {'alpha': self.optimizer.params[0],
                           'tau': self.optimizer.params[1]}, iteration)
        return average_reward

if __name__ == '__main__':
    parser = ArgumentParser()