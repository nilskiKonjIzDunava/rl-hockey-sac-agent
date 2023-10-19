import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple

from utils import create_mirror_masks

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class BasicCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(BasicCritic, self).__init__()
        input_layer_dim = int(state_dim) + int(action_dim)
        # Q1
        self.q1_layer1 = nn.Linear(input_layer_dim, hidden_width)
        self.q1_layer2 = nn.Linear(hidden_width, hidden_width)
        #self.q1_layer3 = nn.Linear(hidden_width, hidden_width)
        self.q1_readout = nn.Linear(hidden_width, 1)

        #Q2
        self.q2_layer1 = nn.Linear(input_layer_dim, hidden_width)
        self.q2_layer2 = nn.Linear(hidden_width, hidden_width)
        #self.q2_layer3 = nn.Linear(hidden_width, hidden_width)
        self.q2_readout = nn.Linear(hidden_width, 1)

      

    def forward(self, s, a):
        state_action = torch.cat([s,a],1)

        q1 = F.relu(self.q1_layer1(state_action))
        q1 = F.relu(self.q1_layer2(q1))
        #q1 = F.relu(self.q1_layer3(q1))
        q1 = self.q1_readout(q1)


        q2 = F.relu(self.q2_layer1(state_action))
        q2 = F.relu(self.q2_layer2(q2))
        #q2 = F.relu(self.q2_layer3(q2))
        q2 = self.q2_readout(q2)

        return q1, q2

class BasicActor(nn.Module):
    def __init__(self, state_dim, action_space, action_dim, hidden_width, reparam_noise):
        super(BasicActor, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space
        self.action_dim = action_dim
        self.reparam_noise = reparam_noise

        self.layer1 = nn.Linear(state_dim, hidden_width)
        self.layer2 = nn.Linear(hidden_width, hidden_width)
        #self.layer3 = nn.Linear(hidden_width, hidden_width)

        self.mu_layer = nn.Linear(hidden_width, action_dim)
        self.log_sigma_layer = nn.Linear(hidden_width, action_dim)
                

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high[:action_dim] - action_space.low[:action_dim]) / 2.).to(self.device)
            self.action_bias = torch.FloatTensor((action_space.high[:action_dim] + action_space.low[:action_dim]) / 2.).to(self.device)

  

    def forward(self, s):
        if not torch.is_tensor(s):
            s = torch.from_numpy(s) 

        s = F.relu(self.layer1(s))
        s = F.relu(self.layer2(s))
        #s = F.relu(self.layer3(s))
        mu = self.mu_layer(s)
        log_sigma = self.log_sigma_layer(s)
        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_sigma


    def sample_policy(self,s : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        s : torch.Tensor, dimension(s) = [batch_size, state_space_dim]

        Returns
        -------
        action : torch.Tensor, dimension(action) = [1, action_space_dim]
            The actions from the policy distribution
        log_policy : torch.Tensor, dimension(log_policy) = [1, 1]
            Log likelihood (log_policy) 
        Parameters of the policy distribution:
        mu :  torch.Tensor, dimension(mu) = [1, action_space_dim]
        sigma :  torch.Tensor, dimension(sigma) = [1, action_space_dim]
        """
        mu, log_sigma = self.forward(s)
        sigma = log_sigma.exp()
        policy = Normal(mu, sigma) # action/policy distribution
        # actions should usually be bounded, 
        # so one has to make sure that sampled actions fall within the bounds
        a = policy.rsample()       # reparametrization trick to make sampling differentiable, gives unbounded actions
        a_squashed = torch.tanh(a) # squash the actions to interval (-1,1)

        action = a_squashed * self.action_scale + self.action_bias # rescale the actions such they fall within bounds of the environment
        
        log_policy = policy.log_prob(a)
        # for numerical stability, see Appendix C of Haarnoja et al. 2019 (https://arxiv.org/abs/1812.05905)
        # add reparam_noise to prevent negative values of the argument of log 
        log_policy -= torch.log(self.action_scale * (1 - a_squashed.pow(2)) + self.reparam_noise) 
        log_policy = log_policy.sum(1, keepdim=True)

        mu = torch.tanh(mu) * self.action_scale + self.action_bias

        return action, log_policy, mu, sigma

class MirrorCritic(BasicCritic):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(MirrorCritic, self).__init__(state_dim, action_dim, hidden_width)

    def forward(self, x, a):
        state_mirror_mask, action_mirror_mask = create_mirror_masks(x)
        x = x * state_mirror_mask
        a = a * action_mirror_mask
        q1, q2 = super().forward(x, a)
        return q1, q2


class MirrorActor(BasicActor):
    def __init__(self, state_dim, action_space, action_dim, hidden_width, reparam_noise):
        super(MirrorActor, self).__init__(state_dim, action_space, action_dim, hidden_width, reparam_noise)

    def forward(self, x):
        state_mirror_mask, action_mirror_mask = create_mirror_masks(x)
        mu, log_sigma = super().forward(x)
        return mu * action_mirror_mask, log_sigma

