import gymnasium as gym
from agent import SACAgent
import numpy as np
from pink_noise import PinkNoiseDist, WhiteNoiseDist


env = gym.make('LunarLander-v2', continuous = True)

score = 0.0
print_interval = 20
max_timesteps = 500 
max_episodes = 10000
state_space_dim = int(env.observation_space.shape[0])
action_space = env.action_space
action_space_dim = int(action_space.shape[0])
agent = SACAgent(state_space_dim,
                action_space,
                action_space_dim,
                use_mirror=False,
                hidden_width=256,
                learning_rate=3e-3,
                learning_rate_alpha= 3e-3,
                discount_rate=0.99,
                batch_size=128,
                max_buffer_size=1e6,
                soft_update_ts=1,
                tau=0.005,
                auto_alpha_tuning=True,
                )
                         
agent.set_training_mode()
score = 0.0   
stop_exploraton_timesteps = 1000
avg_scores = []

for n_epi in range(max_episodes+1):
    s, _ = env.reset()  
    done = False
    count = 0
    load_buffer_type = 'white_noise'
    seq_len = max_timesteps
    if load_buffer_type == 'pink_noise':
        noisy_actions = WhiteNoiseDist(seq_len, agent.action_space_dim).gen.sample(T=seq_len)
    elif load_buffer_type == 'pink_noise':
        noisy_actions = PinkNoiseDist(seq_len, agent.action_space_dim).gen.sample(T=seq_len)
    
    while count < max_timesteps and not done:
        if load_buffer_type == 'white_noise':
            a = env.action_space.sample()
            a = a[:agent.action_space_dim]
            #a = noisy_actions[:, t]
        elif load_buffer_type == 'pink_noise':
            a = noisy_actions[:, count]
        a = agent.act(s.reshape(1,-1))
        #print(a.shape)
        s_prime, r, done, truncated, info = env.step(a.reshape(-1,))
        #print(s_prime.shape)
        agent.replay_buffer.put((s.reshape(-1,), a, r, s_prime.reshape(-1,), done))
        score += r
        s = s_prime
        count += 1

    if agent.replay_buffer.size > stop_exploraton_timesteps: 
        _,_ = agent.train()

    if n_epi%print_interval==0 and n_epi!=0:
        avg_score = score / print_interval
        print("# of episode :{}, avg score : {:.1f}".format(n_epi, avg_score))
        avg_scores.append(avg_score)
        score = 0.0

avg_scores = np.array(avg_scores)
# Generate x-axis values (time steps) for plotting
x_axis = np.arange(print_interval, max_episodes + print_interval, print_interval)
# Combine x-axis values and average scores into a single numpy array
results_data = np.vstack((x_axis, avg_scores))

np.save(load_buffer_type + '_average_scores.npy', results_data)
env.close()
