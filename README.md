# tinygrad-udrl
#### An implementation in [tinygrad](https://github.com/geohot/tinygrad) of upside down reinforcement learning as discussed in [this paper](https://arxiv.org/abs/1912.02875) to the gym environment [Acrobot-v1](https://gym.openai.com/envs/Acrobot-v1/).

Some of the tricks mentioned in the paper were followed like making the training target reward less than the actual reward for that set of actions so the network learns 
to produce actions that will lead to as much or more reward than the target. Another suggestion followed was to tell the network to target a reward twice as good as 
the best its seen so far in twice as much time as it took to accumulate that reward.

The network used was a simple feedforward net with the last 3 states as input (along with the target score and time horizon).

## Results
Overall it's clear that the agent learns to improve in the environment but it usually learns to just spin really fast. This is probably because of the limitations of the network 
used but it could also be that I didn't train it long enough. Either way I got what I wanted out of it (namely practice both turning a paper into code and using tinygrad).

## Dependencies:
 - [numpy](https://numpy.org/)
 - [tinygrad](https://github.com/geohot/tinygrad)
 - [gym](https://gym.openai.com/)
