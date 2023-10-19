import os
import json
import collections
import numpy as np
import laserhockey.hockey_env as h_env

from agent import SACAgent
from logger import Logger
from evaluation import evaluate
from utils import PrioritizedOpponentBuffer
from pink_noise import *

class Trainer:
    """
    A class that implements the training curriculum and saves the models and training metrics.
    """
    def __init__(self):
        # code directory
        self.CODE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
        # output directory for saving the training metrics
        self.OUT_DIR_PATH = os.path.join(self.CODE_DIR_PATH, 'out')
        os.makedirs(self.OUT_DIR_PATH, exist_ok=True)
        # training log file which contains info about the trainings which were done
        # in the past (user configuration)
        self.TRAIN_LOG_FILE = os.path.join(self.OUT_DIR_PATH, 'train_logs.json')
        if not os.path.exists(self.TRAIN_LOG_FILE):
            log_file = open(self.TRAIN_LOG_FILE, 'w')
            json.dump([], log_file, indent=1)
            log_file.close()
        # get the number of the next training run
        training_dirs = [f.name for f in os.scandir(self.OUT_DIR_PATH) if f.is_dir()]
        current_nums = []
        for td in training_dirs:
            if len(td) < 4: continue
            if td.startswith('OUT-') and td[-4:].isnumeric():
                current_nums.append(int(td[-4:]))
        if len(current_nums) < 1: self.CURRENT_INDEX = 0
        else: self.CURRENT_INDEX = int(np.max(current_nums) + 1)
        if self.CURRENT_INDEX == 9999:
            raise Exception('Limit of 9999 files in out reached!')

        
    def train(self, config):
        # initialize the environment
        env_modes = [h_env.HockeyEnv.TRAIN_DEFENSE,
                     h_env.HockeyEnv.TRAIN_SHOOTING,
                     h_env.HockeyEnv.NORMAL]
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)

        # initialize the metrics
        episode_returns = collections.deque([0] * 10000, 10000)     

        # initialize the configs
        
        num_episodes = config['num_episodes']
        max_buffer_size = config['max_buffer_size']
        num_game_ts = config['num_game_ts']
        num_episodes_autosave = config['num_episodes_autosave']
        num_opponent_update = config['num_opponent_update']
        num_beginner_games = config['num_beginner_games']
        wr_opponent_thresh = config['wr_opponent_thresh']
        min_buffer_size_training = config['min_buffer_size_training']
        evaluation_interval = config['evaluation_interval']
        evaluation_ts = config['evaluation_ts']
        print_interval = config['print_interval']
        replay_ratio = config['replay_ratio']
        buffer_noise = config['buffer_noise']
        use_mirror = config['use_mirror']
        discount_rate = config['discount_rate']
        auto_alpha_tuning = config['auto_alpha_tuning']
        alpha = config['alpha']
        learning_rate = config['learning_rate']
        batch_size = config['batch_size']
        hidden_width = config['hidden_width']
        
        """
        #some parameters for faster debugging
        num_episodes = 1000
        max_buffer_size = 50
        num_game_ts = 10
        num_episodes_autosave = 5
        num_opponent_update = 5
        num_beginner_games = 5
        wr_opponent_thresh = 0.00015
        min_buffer_size_training = 10
        evaluation_interval = 8
        evaluation_ts = 5
        print_interval = 5
        replay_ratio = config['replay_ratio']
        buffer_noise = "white_noise"
        use_mirror = True
        discount_rate = 0.95
        auto_alpha_tuning = False
        learning_rate = 1e-3
        batch_size = 128
        alpha = 0.4
        hidden_width = 256
        """
        

        # initialize the main player
        state_space_dim = int(env.observation_space.shape[0])
        action_space = env.action_space
        action_space_dim = int(action_space.shape[0]/2)
    
        main_player = SACAgent(state_space_dim,
                                action_space, 
                                action_space_dim, 
                                use_mirror = use_mirror, 
                                hidden_width=hidden_width,
                                learning_rate=learning_rate,
                                discount_rate = discount_rate,
                                batch_size=batch_size,
                                max_buffer_size=max_buffer_size,
                                auto_alpha_tuning = auto_alpha_tuning,
                                alpha=alpha)
        
        # initialize the opponent buffer
        weak_opponent = h_env.BasicOpponent(weak=True)
        strong_opponent = h_env.BasicOpponent(weak=False) 
        POB = PrioritizedOpponentBuffer(B=1)
        POB.add_opponent(weak_opponent)

        # initialize the opponent
        opponent_idx, opponent = 0, weak_opponent

        # generate noisy action trajectories for populating the replay buffer until it is sufficiently full

        seq_len = num_game_ts

        # episodes loop
        for episode in range(1, num_episodes + 1):
            # change env mode if necessary
            if episode < 2 * num_beginner_games:
                if episode == num_beginner_games:
                    env = h_env.HockeyEnv(mode=env_modes[1])
            else:
                m = np.random.randint(3)
                env = h_env.HockeyEnv(mode=env_modes[m])
                
            # update opponent if necessary
            if POB.K > 1 and episode % num_opponent_update == 0:
                POB.register_outcome(opponent_idx,
                                    -np.mean(list(episode_returns)[-num_opponent_update:]))
                opponent_idx, opponent = POB.get_opponent()

            if buffer_noise == 'white_noise':
                noisy_actions = WhiteNoiseDist(seq_len, main_player.action_space_dim).gen.sample(T=seq_len)
            elif buffer_noise == 'pink_noise':
                noisy_actions = PinkNoiseDist(seq_len, main_player.action_space_dim).gen.sample(T=seq_len)
            elif buffer_noise == 'brown_noise':
                noisy_actions = BrownNoiseDist(seq_len, main_player.action_space_dim).gen.sample(T=seq_len)
                    
            s1, info = env.reset()
            s2 = env.obs_agent_two()
            episode_return = 0

            t = 0
            a1 = np.zeros_like(main_player.action_space.sample()[:main_player.action_space_dim])  

            for t in range(num_game_ts):
                if main_player.replay_buffer.size < min_buffer_size_training:
                    if buffer_noise == 'white_noise':
                        a1 = env.action_space.sample()
                        a1 = a1[:main_player.action_space_dim]
                        #a1 = noisy_actions[:, t]
                    elif buffer_noise == 'pink_noise':
                        a1 = noisy_actions[:, t]
                    elif buffer_noise == 'brown_noise':
                        a1 = noisy_actions[:, t]
                    elif buffer_noise == 'storng_opponent':
                        strong_opp = h_env.BasicOpponent(weak=False)
                        a1 = strong_opp.act(s1)
                else:
                    main_player.set_training_mode() 
                    a1 = main_player.act(s1)
                a2 = opponent.act(s2)
                s_prime, r, done, _, info = env.step(np.hstack([a1, a2]))
                
                main_player.replay_buffer.put((s1, a1, r, s_prime, done))
                episode_return += r
                s1 = s_prime
                s2 = env.obs_agent_two()
                if done:
                    game_outcome = info['winner']
                    break

            episode_returns.append(episode_return)

            if main_player.replay_buffer.size >= min_buffer_size_training: 
                main_player.set_training_mode()
                main_player.train()
                # main_player.train(replay_ratio * t)
            
            if episode % print_interval == 0 and episode != 0:
                avg_return = np.mean(np.array(list(episode_returns)[-print_interval:]))
                self.CURRENT_LOG.register(
                    "# of episode :{}, avg returns : {:.1f}".format(episode, avg_return))
                self.CURRENT_LOG.update_metric('average_returns', episode, avg_return)

            if episode % evaluation_interval == 0 and episode != 0:
                wr_weak = evaluate(main_player, weak_opponent, evaluation_ts, True, self.CURRENT_LOG)
                self.CURRENT_LOG.update_metric('win_rate_weak', episode, wr_weak)
                update_opponent = wr_weak >= wr_opponent_thresh
                if POB.K > 1 and update_opponent:
                    for idx, op in enumerate(POB.opponents[1:]):
                        curr_wr = evaluate(main_player, op, evaluation_ts, True)
                        if idx == 0:
                            self.CURRENT_LOG.update_metric('win_rate_strong', episode, curr_wr)
                        else:
                            self.CURRENT_LOG.update_metric('win_rate_L{idx}', episode, curr_wr)
                        update_opponent &= (curr_wr >= wr_opponent_thresh)
                        if not update_opponent: break
                # add new opponent if the win rate is over some threshold
                # for all current opponents
                if update_opponent:
                    self.CURRENT_LOG.register(f"Episode {episode}: Adding new opponent!\n")
                    if POB.K == 1:
                        # if it is time to update opponent and there is only weak opponent in the buffer add strong opponent
                        POB.add_opponent(strong_opponent)
                    else:
                        new_opponent = SACAgent(state_space_dim,
                                                action_space, 
                                                action_space_dim, 
                                                use_mirror=use_mirror, 
                                                hidden_width=hidden_width,
                                                learning_rate=learning_rate,
                                                discount_rate=discount_rate,
                                                batch_size=batch_size,
                                                max_buffer_size=max_buffer_size,
                                                auto_alpha_tuning=auto_alpha_tuning,
                                                alpha=alpha)
                        new_opponent.actor.load_state_dict(main_player.actor.state_dict())
                        new_opponent.critic.load_state_dict(main_player.critic.state_dict())
                        new_opponent.critic_target.load_state_dict(main_player.critic_target.state_dict())
                      
                        POB.add_opponent(new_opponent)

                    main_player.save_model(f'level-{POB.K - 1}', self.CURRENT_MODELS_DIR)
             
            if episode % num_episodes_autosave == 0:
                main_player.save_model('checkpoint-' + str(episode),
                                       self.CURRENT_MODELS_DIR)

            env.close()

    def start_new_training(self, config):
        # create the directory and update the training log
        # dictionary has the same index as the id of the training in the train_logs.json
        new_dir_name = 'OUT-' + ('0000' + str(self.CURRENT_INDEX))[-4:]       
        self.CURRENT_TRAIN_DIR = os.path.join(self.OUT_DIR_PATH, new_dir_name)
        os.makedirs(os.path.join(self.CURRENT_TRAIN_DIR))
        new_entry = {"id": self.CURRENT_INDEX,
                     "config": config }
        json_file_reader = open(self.TRAIN_LOG_FILE, 'r')
        list_of_trainings = json.load(json_file_reader)
        json_file_reader.close()
        list_of_trainings.append(new_entry)
        json_file_writer = open(self.TRAIN_LOG_FILE, 'w')
        json.dump(list_of_trainings, json_file_writer, indent=1)
        json_file_writer.close()
        # initialize the log of the new training
        #current_logger_path = os.path.join(self.CURRENT_TRAIN_DIR, 'train_logs.txt')
        #self.CURRENT_LOG = Logger(current_logger_path)
        self.CURRENT_LOG = Logger(self.CURRENT_TRAIN_DIR)
        # initialize the models directory
        self.CURRENT_MODELS_DIR = os.path.join(self.CURRENT_TRAIN_DIR, 'models')


        # start training
        self.train(config)
        self.CURRENT_LOG.close()
