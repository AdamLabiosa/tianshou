import gym
import tianshou as ts
import torch, numpy as np
from torch import nn
import argparse
import pprint

from torch.utils.tensorboard import SummaryWriter

DISTRIBUTED = True


parser = argparse.ArgumentParser()
parser.add_argument('--num_nodes', type=int, default=4)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--masterip', type=str, default='10.10.1.1')

args = parser.parse_args()
num_nodes = args.num_nodes
rank = args.rank
masterip = args.masterip

env = gym.make('CartPole-v0')

train_envs = gym.make('CartPole-v0')
test_envs = gym.make('CartPole-v0')

train_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(10)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(100)])

writer = SummaryWriter('log/dqn')
logger = ts.utils.BasicLogger(writer)

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320, distr=DISTRIBUTED, num_nodes=4, rank=rank)

train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True, )
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

try:
    print("hi there")
    _reward = []
    _reward_mean = []
    _std= []
    _time =[]

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=10, step_per_epoch=10, step_per_collect=10,
        update_per_step=0.1, episode_per_test=100, batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
        logger=logger, 
        distributed=DISTRIBUTED,
        num_nodes=num_nodes,
        rank=rank)
    print("=====================================")
    #print("completed!-",i)
    _reward.append(result["best_reward"])
    torch.save(policy.state_dict(), 'dqn.pth')
    #policy.load_state_dict(torch.load('dqn.pth'))
    #pprint.pprint(result)



    
except Exception as e:
    print(e)
    print()
    print("Another process has finished training, exiting...")
    exit()

print("reward mean")
for i in _reward:
    print(i)

print("done!!")
# policy.eval()
# policy.set_eps(0.05)
# collector = ts.data.Collector(policy, env, exploration_noise=True)
# collector.collect(n_episode=1, render=1 / 35)


