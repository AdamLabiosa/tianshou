import gym
import tianshou as ts
import torch, numpy as np
from torch import nn
import argparse
import glob, os

from tianshou.policy import BranchingDQNPolicy
from tianshou.utils.net.common import BranchingNet

# DISTRIBUTED = True
parser = argparse.ArgumentParser()
parser.add_argument('--test-num', type=int)
parser.add_argument('--dist', type=int)
parser.add_argument('--num_nodes', type=int, default=4)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--masterip', type=str, default='10.10.1.1')

args = parser.parse_args()
DISTRIBUTED = bool(args.dist)
test_num = args.test_num
num_nodes = args.num_nodes
rank = args.rank
masterip = args.masterip
print("args:")
print(args)
print()

env = gym.make('CartPole-v0')

train_envs = gym.make('CartPole-v0')
test_envs = gym.make('CartPole-v0')

train_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(10)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(100)])

# class Net(nn.Module):
#     def __init__(self, state_shape, action_shape):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
#             nn.Linear(128, 128), nn.ReLU(inplace=True),
#             nn.Linear(128, 128), nn.ReLU(inplace=True),
#             nn.Linear(128, np.prod(action_shape)),
#         )

#     def forward(self, obs, state=None, info={}):
#         if not isinstance(obs, torch.Tensor):
#             obs = torch.tensor(obs, dtype=torch.float)
#         batch = obs.shape[0]
#         logits = self.model(obs.view(batch, -1))
#         return logits, state

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
# Create branching network with 2 branches

net = BranchingNet(state_shape=state_shape, common_hidden_sizes=[128, 128, 128], value_hidden_sizes=[128, 128, 128], action_hidden_sizes=[128, 128, 128])
optim = torch.optim.Adam(net.parameters(), lr=1e-3)



# policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320, distr=DISTRIBUTED, num_nodes=4, rank=rank)
policy = ts.policy.BranchingDQNPolicy(net, optim, discount_factor=0.9, target_update_freq=320, distr=DISTRIBUTED, num_nodes=4, rank=rank)

train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True, )
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

use_this = True
try:
    result = ts.trainer.offpolicy_trainer(
        test_num,
        policy, train_collector, test_collector,
        max_epoch=10, step_per_epoch=10000, step_per_collect=10,
        update_per_step=0.1, episode_per_test=100, batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold, 
        distributed=DISTRIBUTED,
        num_nodes=num_nodes,
        rank=rank)
except Exception as e:
    use_this = False
    print(e)
    print()
    print("Another process has finished training, exiting...")
    exit()

