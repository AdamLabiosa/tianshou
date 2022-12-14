import gym
import numpy as np
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.policy import A2CPolicy
from tianshou.policy import DDPGPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic
import argparse

import warnings
warnings.filterwarnings('ignore')

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# environments
env = gym.make('Pendulum-v1')
train_envs = DummyVectorEnv([lambda: gym.make('Pendulum-v1') for _ in range(20)])
test_envs = DummyVectorEnv([lambda: gym.make('Pendulum-v1') for _ in range(10)])

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n

# model & optimizer
net = Net(state_shape, hidden_sizes=[64, 64], device=device)
actor = Actor(net, action_shape, device=device).to(device)
critic = Critic(net, device=device).to(device)
actor_critic = ActorCritic(actor, critic)
optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)


dist = torch.distributions.Categorical
# a2c policy
# policy = A2CPolicy(actor=actor, critic=critic, optim=optim, dist_fn=dist, action_space=env.action_space, deterministic_eval=True, distr=DISTRIBUTED, num_nodes=4, rank=rank, masterip=masterip)
# PPO
policy = PPOPolicy(actor, critic, optim, dist, action_space=env.action_space, deterministic_eval=True, distr=DISTRIBUTED, num_nodes=4, rank=rank)
        
          
# collector
train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, len(train_envs)))
test_collector = Collector(policy, test_envs)

# # trainer
# try:
#     result = onpolicy_trainer(
#         policy,
#         train_collector,
#         test_collector,
#         test_num=0,
#         max_epoch=10,
#         step_per_epoch=50000,
#         repeat_per_collect=10,
#         episode_per_test=10,
#         batch_size=256,
#         step_per_collect=2000,
#         stop_fn=lambda mean_reward: mean_reward >= 195,
#     )
# except Exception as e:
#     use_this = False
#     print(e)
#     print()
#     print("Another process has finished training, exiting...")
#     exit()

result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        test_num=0,
        max_epoch=10,
        step_per_epoch=50000,
        repeat_per_collect=10,
        episode_per_test=10,
        batch_size=256,
        step_per_collect=2000,
        stop_fn=lambda mean_reward: mean_reward >= 195,)

print(result)