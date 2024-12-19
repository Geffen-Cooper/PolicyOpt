import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from itertools import product
from sklearn.metrics import f1_score
from experiments.trainer import DeviceTrainer

torch.set_printoptions(sci_mode=True)

class ZerothOrderDeviceTrainer(DeviceTrainer):
    def __init__(self, exp_name, policy_mode, sensor_cfg, train_cfg, classifier_cfg, device, load_path, lr, seed):
        super().__init__(exp_name, policy_mode, sensor_cfg, train_cfg, classifier_cfg, device, load_path, lr, seed)
    
    def _build_optimizer(self, lr):
        # init_params = [1.5e-4, 1e1] # MAX_E - thresh, 
        # init_params = [2e-5, 2e1]
        init_params = [0.0, 9e1]
        # Initialize optimizer
        f = partial(self.sensor.forward_zeroth, training=True)
        self.optimizer = ZerothOrderOptimizer(init_params, lr, self.train_cfg['batch_size'], f, params_bounds=[[0.0, 1.5e-4], [0.0, 100.0]])

    def optimize_model(self, *f_args):
        return self.optimizer.forward(*f_args)

    def train_one_epoch(self, iteration, writer):
        self.sensor.train()
        train_data, train_labels = self.data['train']
        segment_data, segment_labels = self.sensor._sample_segment(train_data, train_labels)
        # add time axis
        t_axis = torch.arange(len(segment_labels), dtype=torch.float64, device=self.device)/self.sensor.FS
        t_axis = t_axis.reshape(-1,1)
        # add the time axis to the data
        train_full_data_window = torch.cat((t_axis, segment_data), dim=1)
        f_args = {
            'data': train_full_data_window,
            'labels': segment_labels
        }
        average_reward = self.optimize_model(f_args)
        
        print("Iteration: {}, average reward: {:.3f}, params: {}, epsilon: {}".format(iteration, average_reward, self.optimizer.params, self.optimizer.epsilon))

        writer.add_scalar("train_metric/average_reward", average_reward, iteration)
        writer.add_scalars("train_metric/params", 
                          {'alpha': self.optimizer.params[0],
                           'tau': self.optimizer.params[1]}, iteration)
        return average_reward

    def train(self):
        writer = SummaryWriter(self.log_dir)
        best_val_reward = 0.0
        best_params = self.optimizer.params

        test_loss = self.test(best_params, *self.data['test'])

        for iteration in tqdm(range(self.train_cfg['epochs'])):
            self.train_one_epoch(iteration, writer)
            if iteration % self.train_cfg['val_every_epochs'] == 0:
                val_loss = self.validate(iteration, writer, *self.data['val'], self.train_cfg['val_iters'])
                if val_loss['avg_reward'] >= best_val_reward:
                    best_params = self.optimizer.params
                    best_val_reward = val_loss['avg_reward']
            
            if self.optimizer.near_convergence == 2:
                print(f"Stopping training as reached convergence with epsilon {self.optimizer.epsilon}")
                print(f"Parameters are {best_params}")
                break
        
        test_loss = self.test(best_params, *self.data['test'])
            
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

                learned_packets, learned_e_trace, actions = self.sensor.forward_sensor(self.optimizer.params, val_full_data_window)
                
                if learned_packets[0] is None:
                    # Policy did not sample at all
                    # print(f"Iteration {iteration}: Policy did not sample at all during validation!")
                    continue

                opp_packets, opp_e_trace, opp_actions = self.sensor.forward_sensor(torch.zeros(2), val_full_data_window, policy_mode="opportunistic") # opportunistic params are [0.0, 0.0]
                

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

        print("Iteration: {}, params: {}, val_policy_f1_score: {:.3f}, val_opp_f1_score {:.3f}".format(iteration, self.optimizer.params, val_policy_f1, val_opp_f1))

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

            learned_packets, learned_e_trace, actions = self.sensor.forward_sensor(params, test_full_data_window)

            if learned_packets[0] is None:
                print("Learned policy did not send during test time")
                return

            opp_packets, opp_e_trace, opp_actions = self.sensor.forward_sensor(torch.zeros(2), test_full_data_window, policy_mode="opportunistic") # opportunistic params are [0.0, 0.0] 

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
            'f1': test_policy_f1,
            'avg_reward': learned_reward,
            'avg_opp_reward': opp_reward,
            'avg_reward_diff': learned_reward - opp_reward,
        }

        if learned_packets[0] is None:
            # Policy did not sample at all
            print(f"Policy did not sample at all during testing so no testing plot!")
            return test_loss
        
        policy_sample_times = (learned_packets[0]).long()
        opp_sample_times = (opp_packets[0]).long()
        self.fig.suptitle(r"$\alpha = {:.3e}, \tau = {:.3e}$".format(params[0], params[1]))
        self.axs.axhline(y=self.sensor.thresh, linestyle='--', color='green') # Opportunistic policy will send at this energy
        self.axs.plot(t_axis, learned_e_trace)
        self.axs.plot(t_axis, opp_e_trace, linestyle='--')
        self.axs.scatter(t_axis[policy_sample_times], learned_e_trace[policy_sample_times], s=100, label='policy')
        self.axs.scatter(t_axis[opp_sample_times], opp_e_trace[opp_sample_times], marker='D', s=100, alpha=0.3,label='opp')
        self.axs.set_xlabel("Time")
        self.axs.set_ylabel("Energy")
        self.axs.legend()
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/plot_test.png")
        self.axs.cla()

        return test_loss
      
class ZerothOrderOptimizer():
    def __init__(self, init_params, epsilon, batch_size, f, params_bounds=None):
        self.params = torch.tensor(init_params)
        self.params_bounds = torch.tensor(params_bounds) if params_bounds is not None else None
        self.epsilon = torch.tensor(epsilon)
        self.batch_size = batch_size
        self.f = f # optimizing wrt first input of f

        self.near_convergence = 0
    
    def _check_params_in_bounds(self, param):
        if self.params_bounds is not None:
            for i,p in enumerate(param):
                if not self.params_bounds[i,0] <= p <= self.params_bounds[i,1]:
                    # print(f"{param} is out of bounds!")
                    return False
            return True
        else:
            return True
    
    def estimate_gradient_and_descent_direction(self, f_args):
        # estimate gradient of f using zeroth-order methods
        # evaluate f(x+\delta_x) for delta_x sampled uniformly. 
        num_params = len(self.params)
        delta_permutations = torch.tensor(list(product([-1,0,1], repeat=num_params)), dtype=torch.float32)
        # delta_permutations = F.normalize(delta_permutations, eps=1.0)
        evaluations = torch.zeros(delta_permutations.shape[0])
        for _ in range(self.batch_size):
            for k, delta in enumerate(delta_permutations):
                if self._check_params_in_bounds(self.params + self.epsilon * delta):
                    param = self.params + self.epsilon * delta
                    evaluations[k] += self.f(param, **f_args) / self.batch_size      
                else:
                    evaluations[k] += -1 # add a negative number since out of bounds
        
        max_value = torch.max(evaluations)
        # Get index of max value. If there are multiple then choose randomly.
        max_index = (evaluations == max_value).nonzero()
        max_index = max_index[torch.randint(low=0, high=len(max_index), size=())].item()
        descent_direction = delta_permutations[max_index]
        
        # if max_value != 0.0:
            # descent_direction *= max_value # max_value \in [0,1] is the mean reward

        return descent_direction, max_value

    def point_update(self, descent_direction):
        if self._check_params_in_bounds(self.params + self.epsilon * descent_direction):
            return self.params + self.epsilon * descent_direction
        else:
            return self.params

    def forward(self, f_args):
        params_before = self.params
        descent_direction, max_value = self.estimate_gradient_and_descent_direction(f_args)
        self.params = self.point_update(descent_direction)

        if torch.equal(params_before, self.params):
            self.epsilon /= 2
            self.near_convergence += 1
            print(f"Decreasing epsilon to {self.epsilon}")
            
        return max_value
    
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
    # lr = [0.5e-4, 1e1]
    lr = [0.5e-4, 1e1]
    policy_mode = "conservative"

    # sensor_cfg = (packet_size, leakage, init_overhead, duration_range, history_size, sample_frequency)
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
        'batch_size': 1,
        'epochs': 5_000,
        'val_iters': 1,
        'val_every_epochs': 1,	
    }

    classifier_cfg = {
        'path': "saved_data/checkpoints/dsads_contig/seed123_activities_[ 0  1  2  3  9 11 15 17 18].pth",
        'num_activities': 9,
    }

    trainer = ZerothOrderDeviceTrainer(exp_name, policy_mode, sensor_cfg, train_cfg, classifier_cfg, device, load_path, lr, seed)
    trainer.train()