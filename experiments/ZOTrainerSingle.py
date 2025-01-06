import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score
from experiments.trainer import DeviceTrainer
from experiments.zero_order_algos import signSGD, SGD, PatternSearch

torch.set_printoptions(sci_mode=True)

class ZOTrainerSingle(DeviceTrainer):
    def __init__(self, exp_name, optimizer_cfg, sensor_cfg, train_cfg, classifier_cfg, device, load_path, data_path, val_user, seed, mode='labeled'):
        self.mode = mode
        super().__init__(exp_name, optimizer_cfg, sensor_cfg, train_cfg, classifier_cfg, device, load_path, data_path, val_user, seed)
        self._preprocess_data()
    
    def _build_optimizer(self):
        # Initialize optimizer
        if self.mode == 'labeled':
            f = partial(self.sensor.forward_zeroth, training=True)
        elif self.mode == 'unlabeled':
            f = partial(self.sensor.forward_zeroth_unlabelled, training=True)
        else:
            raise NotImplementedError(f"Mode {self.mode} is invalid")
        
        Optimizer = self.optimizer_cfg['optimizer']
        self.optimizer = Optimizer(self.optimizer_cfg['init_params'], self.optimizer_cfg['lr'], self.train_cfg['batch_size'], f, params_bounds=self.optimizer_cfg['params_bounds'])
    
    def _preprocess_data(self):
        # TODO
        if self.mode == 'unlabeled':
            train_data = self.data['test']
            val_data = self.data['test']
            test_data = self.data['test']

            self.data = {
                'train': train_data,
                'val': val_data,
                'test': test_data
            }

    def optimize_model(self, *f_args):
        return self.optimizer.forward(*f_args)

    def train_one_epoch(self, iteration, writer, data, labels):
        self.sensor.train()
        segment_data, segment_labels = self.sensor._sample_segment(data, labels)
        # add time axis
        t_axis = torch.arange(len(segment_labels), dtype=torch.float64, device=self.device)/self.sensor.FS
        t_axis = t_axis.reshape(-1,1)
        # add the time axis to the data
        train_full_data_window = torch.cat((t_axis, segment_data), dim=1)

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

    def train(self):
        writer = SummaryWriter(self.log_dir)
        original_epsilon = self.optimizer.epsilon
        best_params = self.optimizer.params

        print(f"Original Epsilon: {original_epsilon}")

        test_loss = self.test(best_params, *self.data['test'])
        best_val_reward = test_loss['policy_reward']

        for iteration in tqdm(range(self.train_cfg['epochs'])):
            self.train_one_epoch(iteration, writer, *self.data['train'])
            if iteration % self.train_cfg['val_every_epochs'] == 0:
                val_loss = self.validate(iteration, writer, *self.data['val'], self.train_cfg['val_iters'])
                if val_loss['avg_reward'] >= best_val_reward:
                    print(f"Saving new best parameters {self.optimizer.params}")
                    best_params = self.optimizer.params
                    best_val_reward = val_loss['avg_reward']
            
            if (self.optimizer.epsilon <= original_epsilon * 1e-2).all():
                print(f"Stopping training as reached convergence with epsilon {self.optimizer.epsilon}")
                print(f"Parameters are {best_params}")
                break
        
        test_loss = self.test(best_params, *self.data['test'])

        return best_params, test_loss
            
    def validate(self, iteration, writer, data, labels, val_iterations):
        self.sensor.eval()
        learned_reward = 0.0
        opp_reward = 0.0
        val_policy_f1 = 0.0
        val_opp_f1 = 0.0

        for _ in range(val_iterations):
            with torch.no_grad():
                segment_data, segment_labels = self.sensor._sample_segment(data, labels)
                t_axis = torch.arange(len(segment_labels), dtype=torch.float64, device=self.device)/self.sensor.FS
                t_axis = t_axis.reshape(-1,1)
                val_full_data_window = torch.cat((t_axis, segment_data), dim=1)

                learned_packets, learned_e_trace, actions = self.sensor.forward_sensor(val_full_data_window, self.optimizer.params)
                
                if learned_packets[0] is None:
                    # Policy did not sample at all
                    # print(f"Iteration {iteration}: Policy did not sample at all during validation!")
                    continue

                opp_packets, opp_e_trace, opp_actions = self.sensor.forward_sensor(val_full_data_window, policy_mode="opportunistic") # opportunistic params are [0.0, 0.0]
                

                outputs_learned, preds_learned, targets_learned = self.sensor.forward_classifier(segment_labels,learned_packets)

                outputs_opp, preds_opp, targets_opp = self.sensor.forward_classifier(segment_labels,opp_packets)

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

        print("Iteration {}: params: {}, val_policy_f1: {:.3f}, val_opp_f1: {:.3f}".format(iteration, self.optimizer.params, val_policy_f1, val_opp_f1))

        if writer is not None:
            writer.add_scalar("val_metric/f1_difference", val_policy_f1 - val_opp_f1, iteration)
            writer.add_scalar("val_metric/policy_f1", val_policy_f1, iteration)
            writer.add_scalar("val_metric/opp_f1", val_opp_f1, iteration)
            writer.add_scalar("val_metric/policy_reward", learned_reward, iteration)
            writer.add_scalar("val_metric/opp_reward", opp_reward, iteration)

        val_loss = {
            'f1': val_policy_f1,
            'avg_reward': learned_reward,
            'avg_reward_diff': learned_reward - opp_reward,
        }

        if learned_packets[0] is None:
            # Policy did not sample at all
            print(f"Iteration {iteration}: Policy did not sample at all during validation so no validation plot!")
            return val_loss
        
        policy_sample_times = (learned_packets[0]).long()
        opp_sample_times = (opp_packets[0]).long()
        self.fig.suptitle(r"$\alpha = {:.3e}, \tau = {:.3e}$".format(self.optimizer.params[0], self.optimizer.params[1]))
        self.axs.plot(t_axis, learned_e_trace)
        self.axs.plot(t_axis, opp_e_trace, linestyle='--')
        self.axs.scatter(t_axis[policy_sample_times], learned_e_trace[policy_sample_times], s=100, label='policy')
        self.axs.scatter(t_axis[opp_sample_times], opp_e_trace[opp_sample_times], marker='D', s=100, alpha=0.3, label='opp')
        self.axs.axhline(y=self.sensor.thresh, linestyle='--', color='green') # Opportunistic policy will send at this energy
        self.axs.set_xlabel("Time")
        self.axs.set_ylabel("Energy")
        self.axs.legend()
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/plot_{iteration}.png")
        self.axs.cla()

        return val_loss

    def test(self, params, data, labels):
        self.sensor.eval()
        learned_reward = 0.0
        opp_reward = 0.0
        test_policy_f1 = 0.0
        test_opp_f1 = 0.0

        with torch.no_grad():
            t_axis = torch.arange(len(labels), dtype=torch.float64, device=self.device)/self.sensor.FS
            t_axis = t_axis.reshape(-1,1)
            test_full_data_window = torch.cat((t_axis, data), dim=1)

            learned_packets, learned_e_trace, actions = self.sensor.forward_sensor(test_full_data_window, params)

            if learned_packets[0] is None:
                print("Learned policy did not send during test time")
                return

            opp_packets, opp_e_trace, opp_actions = self.sensor.forward_sensor(test_full_data_window, policy_mode="opportunistic")

            outputs_learned, preds_learned, targets_learned = self.sensor.forward_classifier(labels,learned_packets)

            outputs_opp, preds_opp, targets_opp = self.sensor.forward_classifier(labels,opp_packets)

            learned_reward += torch.where(preds_learned == targets_learned, 1, 0).sum() / len(preds_learned)
            opp_reward += torch.where(preds_opp == targets_opp, 1, 0).sum() / len(preds_opp)

            test_policy_f1 += f1_score(
                targets_learned.detach().cpu().numpy(), preds_learned.detach().cpu().numpy(), average='macro'
            )
            test_opp_f1 += f1_score(
                targets_opp.detach().cpu().numpy(), preds_opp.detach().cpu().numpy(), average='macro'
            )
        
        num_test_trajs = data.shape[0]
        
        # learned_reward /= num_test_trajs
        # opp_reward /= num_test_trajs
        # test_policy_f1 /= num_test_trajs
        # test_opp_f1 /= num_test_trajs

        print("Test: policy F1: {:.3f}, opportunistic F1 {:.3f}, policy avg reward: {:.3f}, opportunistic avg reward: {:.3f}".format(test_policy_f1, test_opp_f1, learned_reward, opp_reward))

        test_loss = {
            'policy_f1': test_policy_f1,
            'opp_f1': test_opp_f1,
            'policy_reward': learned_reward,
            'opp_reward': opp_reward,
        }

        if learned_packets[0] is None:
            # Policy did not sample at all
            print(f"Policy did not sample at all during testing so no testing plot!")
            return test_loss
        
        policy_sample_times = (learned_packets[0]).long()
        opp_sample_times = (opp_packets[0]).long()
        self.fig.suptitle(r"$\alpha = {:.3e}, \tau = {:.3e}$".format(params[0], params[1]))
        self.axs.plot(t_axis, learned_e_trace)
        self.axs.plot(t_axis, opp_e_trace, linestyle='--')
        self.axs.scatter(t_axis[policy_sample_times], learned_e_trace[policy_sample_times], s=100, label='policy')
        self.axs.scatter(t_axis[opp_sample_times], opp_e_trace[opp_sample_times], marker='D', s=100, alpha=0.3,label='opp')
        self.axs.axhline(y=self.sensor.thresh, linestyle='--', color='green') # Opportunistic policy will send at this energy
        self.axs.set_xlabel("Time")
        self.axs.set_ylabel("Energy")
        self.axs.legend()
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/plot_test.png")
        self.axs.cla()

        return test_loss
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--cross_validation", action='store_true')
    parser.add_argument("--exp_name", type=str, default="ZO")
    args = parser.parse_args()

    exp_name = f"{args.exp_name}_CV" if args.cross_validation else f"{args.exp_name}_SGD_Policy"
    load_path = args.load_path
    data_path = "saved_data/dsads_data/LOOCV_preprocessed_data"
    seed = [0,1,2]
    policy_model = "MLP"
    device = "cpu"
    # lr = [0.5e-4, 1e1]
    # lr = [0.5e-4, 1e1]
    # lr = [1e-5, 5]
    lr = [1e-6, 5] # For PatternSearch
    params_bounds = [[0.0, 1.5e-4], [0.0, 10000.0]]

    optimizer_cfg = {
        'optimizer': PatternSearch, # PatternSearch, signSGD, SGD
        'policy_mode': 'conservative',
        'init_params': [1e-5, 1e2],
        'lr': lr,
        'params_bounds': params_bounds,
    }

    sensor_cfg = {
        'packet_size': 8,
        'leakage': 6e-6,
        'init_overhead': 150e-6,
        'duration_range': (1000,2000),
        'history_size': 16,
        'sample_frequency': 25,
    }

    sensor_net_cfg = {
        'in_dim': 2*sensor_cfg['history_size'],
        'hidden_dim': 32
    }

    sensor_cfg['sensor_net_cfg'] = sensor_net_cfg

    train_cfg = {
        'batch_size': 16,
        'epochs': 5, # 5_000,
        'val_iters': 10,
        'val_every_epochs': 1,	
    }

    classifier_cfg = {
        'path': "saved_data/checkpoints/dsads_contig/seed123_activities_[ 0  1  2  3  9 11 15 17 18].pth",
        'num_activities': 9,
    }

    if args.cross_validation:
        # TODO: generic helper code for LOOCV
        best_params = []
        num_val_users = 8
        policy_f1 = 0.0
        opp_f1 = 0.0
        policy_reward = 0.0
        opp_reward = 0.0
        
        for s in seed:
            for val_user in range(num_val_users):
                trainer = ZOTrainerSingle(exp_name, optimizer_cfg, sensor_cfg, train_cfg, classifier_cfg, device, load_path, data_path, val_user, s)
                params, test_loss = trainer.train()
                best_params.append(params)

                policy_f1 += test_loss['policy_f1']
                opp_f1 += test_loss['opp_f1']
                policy_reward += test_loss['policy_reward']
                opp_reward += test_loss['opp_reward']
        
        policy_f1 /= len(seed) * num_val_users
        opp_f1 /= len(seed) * num_val_users
        policy_reward /= len(seed) * num_val_users
        opp_reward /= len(seed) * num_val_users
        
        print(f"Cross Validation Results over {len(seed)} seeds")
        print("Policy F1: {:.3f}, opportunistic F1 {:.3f}, policy avg reward: {:.3f}, opportunistic avg reward: {:.3f}".format(policy_f1, opp_f1, policy_reward, opp_reward))
        print(f"Best params over the seeds {best_params}")

    else:
        trainer = ZOTrainerSingle(exp_name, optimizer_cfg, sensor_cfg, train_cfg, classifier_cfg, device, load_path, data_path, 0, seed[0], 'labeled')
        params, test_loss = trainer.train()

        unlabeled_optimizer_cfg = {
            'optimizer': PatternSearch,
            'policy_mode': 'conservative',
            'init_params': params,
            'lr': lr,
            'params_bounds': params_bounds,
        }
        unlabeled_exp_name = exp_name + '_val_user_0'

        unlabeled_trainer = ZOTrainerSingle(unlabeled_exp_name, unlabeled_optimizer_cfg, sensor_cfg, train_cfg, classifier_cfg, device, load_path, data_path, 0, seed[0], 'unlabeled')
        unlabeled_params, unlabeled_test_loss = unlabeled_trainer.train()

        new_file_path = os.path.join(unlabeled_trainer.root_dir, "TrialStats.txt")

        with open(new_file_path, "x") as f:
            f.write("Opp F1 {:.3f}, Merged F1 {:.3f}, User F1 {:.3f}".format(test_loss['opp_f1'], test_loss['policy_f1'], unlabeled_test_loss['policy_f1']))
            f.write('\n')
            f.write(f"Init Params {optimizer_cfg['init_params']}, Params Merged {params}, Params User {unlabeled_params}")
        
        print("Opp F1 {:.3f}, Merged F1 {:.3f}, User F1 {:.3f}".format(test_loss['opp_f1'], test_loss['policy_f1'], unlabeled_test_loss['policy_f1']))

        print(f"Init Params {optimizer_cfg['init_params']}, Params Merged {params}, Params User {unlabeled_params}")