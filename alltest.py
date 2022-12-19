import gym
import numpy as np
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# environments
env = gym.make('CartPole-v0')
train_envs = DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(20)])
test_envs = DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(10)])

# model & optimizer
net = Net(env.observation_space.shape, hidden_sizes=[64, 64], device=device)
actor = Actor(net, env.action_space.n, device=device).to(device)
critic = Critic(net, device=device).to(device)
actor_critic = ActorCritic(actor, critic)
optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

# PPO policy
dist = torch.distributions.Categorical
policy = PPOPolicy(actor, critic, optim, dist, action_space=env.action_space, deterministic_eval=True)
        
          
# collector
train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, len(train_envs)))
test_collector = Collector(policy, test_envs)

# trainer
result = onpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    max_epoch=10,
    step_per_epoch=50000,
    repeat_per_collect=10,
    episode_per_test=10,
    batch_size=256,
    step_per_collect=2000,
    stop_fn=lambda mean_reward: mean_reward >= 195,
)
print(result)