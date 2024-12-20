import torch
from abc import ABC, abstractmethod
from itertools import product

class ZeroOrderOptimizer(ABC):
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
    
    @abstractmethod
    def estimate_gradient_and_descent_direction(self, f_args):
        pass

    @abstractmethod
    def point_update(self, descent_direction):
        pass

    @abstractmethod
    def forward(self, f_args):
        pass

class SGD(ZeroOrderOptimizer):
    def __init__(self, init_params, epsilon, batch_size, f, params_bounds=None):
        super().__init__(init_params, epsilon, batch_size, f, params_bounds)

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
        
        if max_value != 0.0:
            descent_direction *= max_value # max_value \in [0,1] is the mean reward

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
    

class signSGD(SGD):
    def __init__(self, init_params, epsilon, batch_size, f, params_bounds=None):
        super().__init__(init_params, epsilon, batch_size, f, params_bounds)
    
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

        return descent_direction, max_value