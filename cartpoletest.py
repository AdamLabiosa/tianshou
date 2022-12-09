import gym
import numpy as np
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--num_nodes', type=int, default=4)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--masterip', type=str, default='10.10.1.1')

if __name__ == '__main__':
    args = parser.parse_args()
    num_nodes = args.num_nodes
    rank = args.rank
    masterip = args.masterip

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('CartPole-v0')
    train_envs = DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(20)])
    test_envs = DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(10)])

    net = Net(env.observation_space.shape, hidden_sizes=[64, 64], device=device)
    actor = Actor(net, env.action_space.n, device=device).to(device)
    critic = Critic(net, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)
    

    # PPO policy
    dist = torch.distributions.Categorical
    policy = PPOPolicy(actor, critic, optim, dist, action_space=env.action_space, deterministic_eval=True, distribute=True, num_nodes=num_nodes, rank=rank, masterip=masterip)
            
            
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
        distributed=True,
        num_nodes=4,
        rank=0,
    )
    print(result)

    # # Let's watch its performance!
    # policy.eval()
    # result = test_collector.collect(n_episode=1, render=True)
    # print("Final reward: {}, length: {}".format(result["rews"].mean(), result["lens"].mean()))