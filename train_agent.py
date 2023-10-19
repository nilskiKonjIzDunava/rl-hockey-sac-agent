from optparse import OptionParser
from trainer import Trainer
import json

parser = OptionParser()
parser.add_option("--num_episodes", dest="num_episodes", default=25000, help="number of episodes", metavar="N")
parser.add_option("--max_buffer_size", dest="max_buffer_size", default=1e5, help="the size of the replay buffer", metavar="N")
parser.add_option("--num_game_ts", dest="num_game_ts", default=250, help="number of maximum time steps per episode", metavar="N")
parser.add_option("--num_episodes_autosave", dest="num_episodes_autosave", default=1000, help="number of episodes after which autosave is performed", metavar="N")
parser.add_option("--num_opponent_update", dest="num_opponent_update", default=50, help="number of episodes after which we query the POB for the new opponent", metavar="N")
parser.add_option("--num_beginner_games", dest="num_beginner_games", default=2500, help="number of initial episodes in which it is trained using a simple curriculum", metavar="N")
parser.add_option("--wr_opponent_thresh", dest="wr_opponent_thresh", default=1.01, help="number of maximum time steps per episode", metavar="TRSH")
parser.add_option("--min_buffer_size_training", dest="min_buffer_size_training", default=2000, help="the size of replay buffer required to start training", metavar="N")
parser.add_option("--evaluation_interval", dest="evaluation_interval", default=500, help="number of episodes after which the evaluation on weak agent should start", metavar="N")
parser.add_option("--evaluation_ts", dest="evaluation_ts", default=200, help="number of time steps per evaluation", metavar="N")
parser.add_option("--print_interval", dest="print_interval", default=100, help="number of episodes after which the average returns in last N episodes should be printed", metavar="N")
parser.add_option("--replay_ratio", dest="replay_ratio", default=1, help="replay ratio", metavar="N")
parser.add_option("--buffer_noise", dest="buffer_noise", default="pink_noise",
                        choices=["white_noise", "pink_noise", "brown_noise", "strong_opponent"],
                        help="Choose the action noise to populate the buffer first min_buffer_size_training environment steps: white_noise, pink_noise, brown_noise, strong_opponent")
parser.add_option("--use_mirror", dest="use_mirror", default=True, help="Whether to learn the mirrored actions", metavar="TYPE")
parser.add_option("--discount_rate", dest="discount_rate", default=0.95, type = float, help="determines how much the reinforcement learning agents cares about rewards in the distant future relative to those in \
                  the immediate future. If discout rate = 0, the agent will be completely myopic and only learn about actions that produce an immediate reward.", metavar="N")
parser.add_option("--auto_alpha_tuning", dest="auto_alpha_tuning", default=True, help="Whether to use automatic temperature tuning", metavar="TYPE")
parser.add_option("--alpha", dest="alpha", default=0.1, help="temperature parameter when automatic tuning is off", metavar="N")
parser.add_option("--learning_rate", dest="learning_rate", default=1e-3, help="learning rate used for optimizing actor and acritic", metavar="N")
parser.add_option("--batch_size", dest="batch_size", default=128, help="number of transitions to take from replay buffer for learning", metavar="N")
parser.add_option("--hidden_width", dest="hidden_width", default=256, help="the width of the hidden layer for the actor and critic networks", metavar="N")

def main():
    (options, args) = parser.parse_args()
    config = {
        "num_episodes": int(options.num_episodes),
        "max_buffer_size": int(options.max_buffer_size),
        "num_game_ts": int(options.num_game_ts),
        "num_episodes_autosave": int(options.num_episodes_autosave),
        "num_opponent_update": int(options.num_opponent_update),   
        "num_beginner_games" :int(options.num_beginner_games), 
        "wr_opponent_thresh": float(options.wr_opponent_thresh),
        "min_buffer_size_training": int(options.min_buffer_size_training),
        "evaluation_interval": int(options.evaluation_interval),
        "evaluation_ts": int(options.evaluation_ts),
        "print_interval": int(options.print_interval),
        "replay_ratio": int(options.replay_ratio),
        "buffer_noise": options.buffer_noise,
        "use_mirror": options.use_mirror,
        "discount_rate" : options.discount_rate,
        "auto_alpha_tuning": options.auto_alpha_tuning,
        "alpha" : options.alpha,
        "learning_rate" : options.learning_rate,
        "batch_size" : options.batch_size,
        "hidden_width" : options.hidden_width
        }
    
    trainer = Trainer()
    trainer.start_new_training(config)


if __name__ == "__main__":
    main()
