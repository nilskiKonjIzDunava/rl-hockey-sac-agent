import numpy as np
import collections
import torch
import numpy.typing as npt
from typing import Tuple

# opponent sampling based on Sliding-Window UCB
class PrioritizedOpponentBuffer():
    """
    A class that selects the opponents during training. This problem can be then formulated as a multi-armed bandit problem where the opponents are the arms and their expected
    rewards change over the time so the problem is non-stationary. In order to solve this problem, Discounted UCB (D-UCB) is used.
    """
    def __init__(self, B=1, xi=1, gamma=.95, tau=None):
        self.B = B
        if tau is None: self.tau = min(1e3, int(np.log(1e-2) / np.log(gamma))) # the solution of inequality  gamma**tau < 0.01, so the discount rate is prevented from going to 0
        else: self.tau = tau
        self.xi = 1
        self.gamma = gamma
        self.opponents = []
        self.history = collections.deque(self.tau * [-1], self.tau)
        self.history_outcomes = collections.deque(self.tau * [-1], self.tau)
        self.t = 0 # index of the opponent in the 
        self.K = 0 # number of opponents in the POB

    def add_opponent(self, opponent):
        self.opponents.append(opponent)
        self.K = len(self.opponents)
        self.t = min(self.t, self.tau)
        
    def get_opponent(self):
        if self.K < 1:
            print('The buffer is empty!')
            return None
        if self.t < self.K:
            opponent = self.opponents[self.t]
            return self.t, opponent
        else:
            # a matrix of dimensions K x N, where K is a number of unique opponents (arms) that the agent played with 
            # and N is a total number of opponents that the agent played with (time step)
            opponent_history = (self.history == np.arange(self.K).reshape(-1, 1)).astype(int)
            discount = (self.gamma ** np.arange(self.tau))[::-1] # for sliding discounted ucb
            N = np.sum(opponent_history * discount + 1e-6, axis=1)
            X = np.sum(opponent_history * self.history_outcomes * discount, axis=1) / N
            c = 2 * self.B * np.sqrt(self.xi * np.log(np.sum(N)) / N) # a padding function
            final = np.nan_to_num(X + c, copy=False, nan=np.inf).flatten()
            opponent_idx = np.argmax(final)
            opponent = self.opponents[opponent_idx]
            return opponent_idx, opponent
        
    def register_outcome(self, opponent_idx, outcome):
        #outcome - mean value of episode returns over past num_opponent_update time steps
        self.history.append(opponent_idx)
        self.history_outcomes.append(outcome)
        self.t += 1

        
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        state_dim = int(state_dim)
        action_dim = int(action_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.next_state = np.zeros((self.max_size, state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))


    def put(self, transition : Tuple[npt.NDArray, npt.NDArray, float, npt.NDArray, bool]) -> None:
        state, action, reward, next_state, done = transition
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size : int) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu, dtype=np.float32)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        x = np.float32(x)
        self.x_prev = x
        return x


# exploiting mirroring
# multiply state and action spaces y coordinates with -1
# mirror(a) := a[1, 2] *= -1
# mirror(s) := s[1, 2, 4, 5, 7, 8, 10, 11, 13, 15] *= -1
def create_mirror_masks(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mirror = (x[..., 1] < 0)
    mirror_new_shape = list(x.shape)
    mirror_new_shape[-1] = 1
    mirror = (-2 * mirror.reshape(tuple(mirror_new_shape))) + 1
    # avoiding inplace operations, since tensors should be differentiable
    double_mirror = torch.hstack((mirror, mirror))
    em = torch.ones(mirror.shape).to(device)
    state_mirror_mask = torch.hstack((em, double_mirror, em, double_mirror, em, double_mirror, em, double_mirror, em, mirror, em, mirror, em, em))
    action_mirror_mask = torch.hstack((em, double_mirror, em))
    return state_mirror_mask, action_mirror_mask
