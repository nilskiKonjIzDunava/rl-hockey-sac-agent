# SAC agent in a Laser Hockey Environment

This repository contains the implementation of the SAC agent designed for the [HockeyEnv](https://github.com/martius-lab/laser-hockey-env). This is the challenging custom environment made using the [OpenAI Gymnasium](https://gymnasium.farama.org/index.html)
and the Box2D engine. It is a two-player game where the agent has to learn to defend its goal, handle the puck, and shoot it to the opponent's goal. To learn these skills the agent undergoes a curriculum learning by utilizing three play modes (defense, shoot, normal).
It levels up its play in the tournament firstly against a weak opponent, then a strong opponent, which are provided by the Hockey Environment. In order not to overfit its play to the weak and strong opponents, when it reaches a certain win rate against them, it plays
with the previous versions of itself. This is facilitated by the Prioritized Opponent Buffer, formulated as a multi-armed bandit problem. 
To impove the early state space exploration, in addition to taking random actions, the agent can also take the actions sampled from the pink noise or brown noises. 
For more details about the implementation and the choice of hyperparameters see the project report and the presentation in the [assets folder](https://github.com/nilskiKonjIzDunava/rl-hockey-sac-agent/tree/main/assets).

Additionally, the implementation can be checked in the OpenAI's [Lunar Lander environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/), which is more convenient for debugging purposes.

