"""
implementation of replay memory for the agent
"""
import torch
import torch.nn as nn

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import deque

class Memory():
    def __init__(self, obs_shape, num_steps, num_env, device, gamma=.999, lam=.95, normalize_adv = True):
        self.obs_shape = obs_shape
        self.num_steps = num_steps 
        self.num_env =  num_env
        self.device = device 
        self.gamma = gamma 
        self.lam = lam 
        self.normalize_adv = normalize_adv 
        self.reset()

    def reset(self):
        self.obs = torch.zeros(self.num_steps +1, self.num_env, *self.obs_shape)
        self.action = torch.zeros(self.num_steps, self.num_env)
        self.reward = torch.zeros(self.num_steps, self.num_env)
        self.done = torch.zeros(self.num_steps, self.num_env)
        self.deltas = torch.zeros(self.num_steps, self.num_env)
        self.returns = torch.zeros(self.num_steps, self.num_env)
        self.advantage = torch.zeros(self.num_steps, self.num_env)
        self.info = deque(maxlen=self.num_steps)
        self.value = torch.zeros(self.num_steps+1, self.num_env)
        self.step = 0

    def store(self, obs, action, reward, done, info, value):
        self.obs[self.step] = obs.clone()
        self.action[self.step] = action.clone()
        self.reward[self.step] = torch.from_numpy(reward.copy())
        self.done[self.step] = torch.from_numpy(done.copy())
        self.info.append(info)
        self.value[self.step] = value.clone()
        self.step = (self.step + 1) % self.num_steps

    def store_last(self, obs, value):
        self.obs[-1] = obs.clone()
        self.value[-1] = value.clone()

    def compute_return_advantage(self):
        advantage = 0
        future_reward = 0

        for i in reversed(range(self.num_steps)):
            mask = 1 - self.done[i]
            future_reward = self.reward[i] + self.gamma * future_reward * mask
            delta = self.reward[i] + self.gamma * self.value[i + 1] * mask - self.value[i]

            advantage = self.gamma * self.lam* advantage * mask + delta
            self.returns[i] = future_reward
            self.deltas[i] = delta
            self.advantage[i] = advantage

        if self.normalize_adv:
            self.advantage = (self.advantage - self.advantage.mean()) / (
                self.advantage.std() + 1e-9
            )

    def get_generator(self, batch_size=1024):
        iterator = BatchSampler(
            SubsetRandomSampler(range(self.num_steps * self.num_env)), batch_size, drop_last=True
        )
        for indices in iterator:
            obs = self.obs[:-1].reshape(-1, *self.obs_shape)[indices].to(device=self.device)
            action = self.action.reshape(-1)[indices].to(device=self.device)
            value = self.value[:-1].reshape(-1)[indices].to(device=self.device)
            returns = self.returns.reshape(-1)[indices].to(device=self.device)
            delta = self.deltas.reshape(-1)[indices].to(device=self.device)
            advantage = self.advantage.reshape(-1)[indices].to(device=self.device)
            yield obs, action, returns, delta, advantage ,value

    def get_reward(self, normalized_reward=True):
        if normalized_reward:
            reward = []
            for step in range(self.num_steps):
                info = self.info[step]
                reward.append([d["reward"] for d in info])
            reward = torch.Tensor(reward)
            
        else:
            reward = self.reward

        return reward.sum(0)
