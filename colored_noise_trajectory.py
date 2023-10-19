
import numpy as np
import matplotlib.pyplot as plt
import os
import laserhockey.hockey_env as h_env
from pink_noise import PinkNoiseDist, BrownNoiseDist, WhiteNoiseDist
from agent import SACAgent
import scienceplots

def save_plot(save_plot_path='./plots/'):
    save_plot_path='./plots/'
    if not os.path.exists(save_plot_path): os.makedirs(save_plot_path)


def generate_cplored_noise_series():
    rng = np.random.default_rng(0)
    seq_len = 250  
    action_space_dim = 4

    # Generate pink noise samples
    s_batch_pink_noise = PinkNoiseDist(seq_len, action_space_dim, rng).gen.sample(T=seq_len)
    s_batch_brown_noise = BrownNoiseDist(seq_len, action_space_dim, rng).gen.sample(T=seq_len)
    s_batch_white_noise = WhiteNoiseDist(seq_len, action_space_dim, rng).gen.sample(T=seq_len)
    pink_trajectory_data = s_batch_pink_noise[0, :]
    brown_trajectory_data = s_batch_brown_noise[0, :]
    white_trajectory_data = s_batch_white_noise[0, :]

    plt.plot(np.asarray(range(seq_len)), pink_trajectory_data, '-m')
    plt.plot(np.asarray(range(seq_len)), brown_trajectory_data, '-b')
    plt.plot(np.asarray(range(seq_len)), white_trajectory_data, '-g')
    plt.xlabel("Environment step")
    plt.ylabel("Action")
    plt.show()

    save_plot_path='./plots/'
    if not os.path.exists(save_plot_path): os.makedirs(save_plot_path)
    save_plot_path = os.path.join(save_plot_path, 'time_domain_noise.png')
    plt.savefig(save_plot_path)  # Save the plot

def plot_multi_episode_states(all_episode_states, load_buffer_type):
    with plt.style.context(['science', 'nature', 'notebook']):
        plt.figure(figsize=(6.4, 4.8))
        
        for episode_num, episode_states in enumerate(all_episode_states, start=1):
            episode_states = np.asarray(episode_states)
            plt.plot(episode_states[:, 0], episode_states[:, 1], label=f'Episode {episode_num}')

        plt.xlabel("x")
        plt.ylabel("y")
        if load_buffer_type == 'white_noise':
            title  = "Visited states over episodes with the white noise actions"
        elif load_buffer_type == 'pink_noise':
            title  = "Visited states over episodes with the pink noise actions"
        elif load_buffer_type == 'brown_noise':
            title  = "Visited states over episodes with the brown noise actions"
        elif load_buffer_type == 'storng_opponent':
            title  = "Visited states over episodes with the strong oponent's actions"
        plt.title(title)
        #plt.legend(loc = 'lower right')

    save_plot_path = './plots/diff_size/'
    if not os.path.exists(save_plot_path):
        os.makedirs(save_plot_path)
    save_plot_path = os.path.join(save_plot_path, load_buffer_type + '_all_episodes_states.png')
    plt.savefig(save_plot_path)  # Save the plot
    plt.close()  # Close the figure

def plot_multi_episode_actions(all_episode_actions, load_buffer_type):
    with plt.style.context(['science', 'nature', 'notebook']):
        plt.figure(figsize=(12, 8))
        
        for episode_num, episode_actions in enumerate(all_episode_actions, start=1):
            episode_actions = np.asarray(episode_actions)
            plt.plot(np.asarray(range(episode_actions.shape[0])), episode_actions, label=f'Episode {episode_num}')

        plt.xlabel("Time Step")
        plt.ylabel("Action - change of the position in x direction")
        if load_buffer_type == 'white_noise':
            title  = "White noise actions over episodes"
        elif load_buffer_type == 'pink_noise':
            title  = "Pink noise actions over episodes"
        elif load_buffer_type == 'brown_noise':
            title  = "Brown noise actions over episodes"
        elif load_buffer_type == 'storng_opponent':
            title  = "Strong oponent's actions"
        plt.title(title)
        #plt.legend(loc = 'lower right')

    save_plot_path = './plots/'
    if not os.path.exists(save_plot_path):
        os.makedirs(save_plot_path)
    save_plot_path = os.path.join(save_plot_path, load_buffer_type + '_all_episodes_actions.png')
    plt.savefig(save_plot_path)  # Save the plot
    plt.close()  # Close the figure



def plot_trajectories(p1, p2, load_buffer_type, seq_len = 250, n_duels=20):
    p1.set_evaluation_mode()

    all_episode_states = []
    all_episode_actions = []

    for test_game in range(n_duels):
        env = h_env.HockeyEnv()

        if load_buffer_type == 'white_noise':
            noisy_actions = WhiteNoiseDist(seq_len, p1.action_space_dim).gen.sample(T=seq_len)
        elif load_buffer_type == 'pink_noise':
            noisy_actions = PinkNoiseDist(seq_len, p1.action_space_dim).gen.sample(T=seq_len)
        elif load_buffer_type == 'brown_noise':
            noisy_actions = BrownNoiseDist(seq_len, p1.action_space_dim).gen.sample(T=seq_len)

        episode_states = []
        episode_actions = []

        s1, info = env.reset()
        #print("state dim", s1[:2].shape)
        episode_states.append(s1[:2])
        s2 = env.obs_agent_two()
        a1 = np.zeros_like(p1.action_space.sample()[:p1.action_space_dim])
        for t in range(250):
            if load_buffer_type == 'white_noise':
                a1 = env.action_space.sample()
                a1 = a1[:p1.action_space_dim]
                #a1 = noisy_actions[:, t]
            elif load_buffer_type == 'pink_noise':
                a1 = noisy_actions[:, t]
            elif load_buffer_type == 'brown_noise':
                a1 = noisy_actions[:, t]
            elif load_buffer_type == 'storng_opponent':
                strong_opp = h_env.BasicOpponent(weak=False)
                a1 = strong_opp.act(s1)
            episode_actions.append(a1[0])
            #a1 = p1.act(s1)
            a2 = p2.act(s2)
            s1, r, done, _, _ = env.step(np.hstack([a1, a2]))
            episode_states.append(s1[:2])
            s2 = env.obs_agent_two()
            if done:
                all_episode_states.append(episode_states)
                all_episode_actions.append(episode_actions)
                break
        env.close()
    plot_multi_episode_states(all_episode_states, load_buffer_type)
    plot_multi_episode_actions(all_episode_actions, load_buffer_type)
        
    


def main():
    env = h_env.HockeyEnv()
    state_space_dim = int(env.observation_space.shape[0])
    action_space = env.action_space
    action_space_dim = int(action_space.shape[0]/2)
    main_player = SACAgent(state_space_dim,
                                    action_space, 
                                    action_space_dim)
    weak_opponent = h_env.BasicOpponent(weak=True)

    plot_trajectories(main_player, weak_opponent, 'white_noise', 250, 10)

if __name__ == "__main__":
    main()