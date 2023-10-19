import os
import torch
import torch.nn.functional as F
from pink_noise import *
import copy

from utils import ReplayBuffer
from networks import BasicActor, BasicCritic, MirrorActor, MirrorCritic


class SACAgent():
    def __init__(self, 
                 state_space_dim,
                 action_space,
                 action_space_dim, 
                 use_mirror = True,
                 hidden_width = 256,
                 learning_rate = 1e-3,
                 discount_rate = 0.95,
                 batch_size = 128,
                 max_buffer_size = 1e5,
                 soft_update_ts = 1,
                 tau = 0.005, 
                 reparam_noise = 1e-6,
                 auto_alpha_tuning = True,
                 learning_rate_alpha = 1e-4,
                 alpha = 0.1
                ):
        
        # set SAC parameters
        self.identifier = "SAC"
        self.state_space_dim = state_space_dim
        self.action_space = action_space
        self. action_space_dim = action_space_dim
        self.hidden_width = hidden_width
        self.learning_rate = learning_rate #for both actor and critic
        self.discount_rate = discount_rate
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size
        self.tau = tau
        self.reparam_noise = reparam_noise
        self.auto_alpha_tuning = auto_alpha_tuning
        self.learning_rate_alpha = learning_rate_alpha
        self.alpha = alpha                                          
        self.soft_update_ts = soft_update_ts
        
        self.last_soft_update = 0
        self.time_step = 0
        self.evaluation_mode = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Critic, Actor = BasicCritic, BasicActor
        if use_mirror:
            Critic = MirrorCritic
            Actor = MirrorActor

        # make actor and critic components
        self.actor = Actor(self.state_space_dim, self. action_space, self.action_space_dim, self.hidden_width, reparam_noise).to(device=self.device)
        self.critic = Critic(self.state_space_dim, self.action_space_dim, self.hidden_width).to(self.device)
        self.critic_target = Critic(self.state_space_dim, self.action_space_dim, self.hidden_width).to(self.device)
        
        self.actor_optimizer = torch.optim.NAdam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.NAdam(self.critic.parameters(), lr=self.learning_rate)
      

        self.critic_target.load_state_dict(self.critic.state_dict())                                                                        

        # set optimizer for temperature parameter alpha 
        if auto_alpha_tuning:
            self.target_entropy = -torch.tensor(self.action_space_dim).to(self.device)                                                 
            self.log_alpha = torch.zeros(1, requires_grad=True, device = self.device)
            self.alpha_optimizer = torch.optim.NAdam([self.log_alpha], lr=self.learning_rate_alpha)
       
        # make replay buffer
        self.replay_buffer = ReplayBuffer(self.state_space_dim, self.action_space_dim, self.max_buffer_size)


    # define SAC agent's functions for training

    def calculate_td_target(self, r, s_next, d) -> torch.Tensor:
        """
        Calculates the targets for the Q functions.

        Parameters
        ----------
        r : torch.Tensor, dimension(s) = [batch_size, 1]
        s_next : torch.Tensor, dimension(s) = [batch_size, state_space_dim]
        d : torch.Tensor, dimension(s) = [batch_size, 1]

        Returns
        -------
        td_target : torch.Tensor, dimensions = [batch_size, 1] 
            Targets for the Q functions
        """
        with torch.no_grad():
            a_next_policy, log_policy_next,_,_ = self.actor.sample_policy(s_next)
            q1_targ_next, q2_target_next = self.critic_target.forward(s_next, a_next_policy)
            q_target_min = torch.min(q1_targ_next, q2_target_next)
            entropy = -self.alpha*log_policy_next
            temp = q_target_min + entropy
            td_target = r + self.discount_rate*d*temp
        return td_target
    

    def optimize_critic(self, s : torch.Tensor, a : torch.Tensor, r : torch.Tensor, s_next : torch.Tensor, d : torch.Tensor) -> torch.Tensor:
        """
        Preforms minimization of the critic objective function.

        Parameters
        ----------
        s : torch.Tensor, dimension(s) = [batch_size, state_space_dim]
        a : torch.Tensor, dimension(a) = [batch_size, action_space_dim]
        r : torch.Tensor, dimension(r) = [batch_size, 1]
        s_next : torch.Tensor, dimension(s_next) = [batch_size, state_space_dim]
        d : torch.Tensor, dimension(d) = [batch_size, 1]

        Returns
        -------
        critic_loss : torch.Tensor, dimension(critic_loss) = [] (scalar value)
            A critic loss
        """
        td_target = self.calculate_td_target(r, s_next, d)
        q1, q2 = self.critic(s,a)
        critic_1_loss = F.mse_loss(q1, td_target.detach()) 
        critic_2_loss = F.mse_loss(q2, td_target.detach())
        critic_loss = critic_1_loss + critic_2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.detach()


    def optimize_actor(self, s : torch.Tensor) -> torch.Tensor:
        """
        Preforms minimization of the actor objective function.

        Parameters
        ----------
        s : torch.Tensor, dimension(s) = [batch_size, state_space_dim]

        Returns
        -------
        policy_loss : torch.Tensor, dimension(policy_loss) = [] (scalar value)
            An actor loss
        alpha_loss : torch.Tensor, dimension(policy_loss) = [] (scalar value)
            The temperatue loss
        """
        a_policy, log_policy, _ , _ = self.actor.sample_policy(s)
        #entropy = -self.log_alpha.exp() * log_policy
        entropy = self.alpha * log_policy

        q1_policy, q2_policy = self.critic.forward(s, a_policy)
        min_q_policy = torch.min(q1_policy, q2_policy)
        policy_loss = (self.alpha*entropy - min_q_policy).mean(axis=0)

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # adjust the temperature
        if self.auto_alpha_tuning:
            # alpha has to be greater than 0 => optimize log_alpha to omit the lower bound (log_alpha can take any real value - there is no constraint)
            alpha_loss = -(self.log_alpha * (log_policy + self.target_entropy).detach()).mean() # detach variables that we don't want to optimize
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
        return policy_loss.detach(), alpha_loss.detach()
    

    def set_evaluation_mode(self):
        self.evaluation_mode = True

    def set_training_mode(self):
        self.evaluation_mode = False

    def act(self, observations):
        """
        (Called by agent)
        Returns the action depending on the mode of the agent 
        """
        if self.evaluation_mode:
            action = self._act(observations, True) 
        else:
            action = self._act(observations, False)
        return action

    def _act(self, observations, evaluate = False):
        """
        (For internal use, called by act(self, observations))
        Returns the action depending on the mode of the agent:
        - training mode: get the action sampled from policy
        - evaluation mode: when exploiting the policy at the test time take MAP of the policy distribution to remove the noise and see what was learned
        """
        s = torch.FloatTensor(observations).to(self.device).unsqueeze(0) 
        if evaluate is False:
            action, _, _, _ = self.actor.sample_policy(s)
        else:
            _, _, action, _ = self.actor.sample_policy(s) 
        return action.detach().cpu().numpy()[0]
    

    def train(self,fit_iter : int =32) -> list[float]:
        actor_losses, critic_losses = [], []
        for t in range(fit_iter):
            s_batch,a_batch,r_batch,s_next_batch,d_batch  = self.replay_buffer.sample(self.batch_size) # returns a touple of tensors
                
            critic_losses.append(self.optimize_critic(s_batch, a_batch, r_batch, s_next_batch, d_batch).detach())

            policy_loss, alpha_loss = self.optimize_actor(s_batch)
            actor_losses.append(policy_loss.detach())

            self.time_step += 1
            if self.time_step - self.last_soft_update == self.soft_update_ts:
                self.soft_update(self.critic, self.critic_target)
                self.last_soft_update = self.time_step
        return actor_losses, critic_losses

    def soft_update(self, net, net_target):
        """
        Updates the target networks by Polyak averaging.
        """
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        

            
    def load_models(self, actor_model_path, critic_model_path):
        """
        Loads the trained models.
        """
        self.actor.load_state_dict(torch.load(actor_model_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_model_path, map_location=self.device))



    def save_model(self, name, save_model_path='./models/'):
        actor_path = os.path.join(save_model_path, 'actor')
        critic_path = os.path.join(save_model_path, 'critic')
        if not os.path.exists(actor_path): os.makedirs(actor_path)
        if not os.path.exists(critic_path): os.makedirs(critic_path)
        actor_path = os.path.join(actor_path, name + '.pt')
        critic_path = os.path.join(critic_path, name + '.pt')
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    
    
    def remote_act(self, state):
        return self.act(state)

    def before_game_starts(self) -> None:
        pass

    def after_game_ends(self) -> None:
        pass

