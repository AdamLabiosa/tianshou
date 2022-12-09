import gym
import numpy as np
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import BCQPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import BranchingNet, Net
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

    '''state_shape: Union[int, Sequence[int]],
        num_branches: int = 0,
        action_per_branch: int = 2,
        common_hidden_sizes: List[int] = [],
        value_hidden_sizes: List[int] = [],
        action_hidden_sizes: List[int] = [],
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,'''

    net = BranchingNet(state_shape=env.observation_space.shape, num_branches=1, action_per_branch=env.action_space.n, common_hidden_sizes=[64, 64], value_hidden_sizes=[64, 64], action_hidden_sizes=[64, 64], device=device).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=0.0003)
    

    # PPO policy
    dist = torch.distributions.Categorical
    policy = BCQPolicy(net, optim, dist, action_space=env.action_space, deterministic_eval=True, distr=True, num_nodes=num_nodes, rank=rank, masterip=masterip)
            
            
    # collector
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # off policy trainer
    result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            max_epoch=10    ,
            step_per_epoch=50000,
            repeat_per_collect=10,
            episode_per_test=10,
            batch_size=256,
            step_per_collect=2000,
            stop_fn=lambda mean_reward: mean_reward >= 195,
            distributed=True,
            num_nodes=num_nodes,
            rank=rank,
    )

    print(result)