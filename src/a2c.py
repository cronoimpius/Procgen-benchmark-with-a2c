"""
implementatio of actor critic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class A2C(nn.Module):
    def __init__(self, model, in_dim, actions, grad_eps, device, beta_v=.5, beta_e=.01):
        super().__init__()
        self.model = model 
        self.actor = nn.Linear(in_dim, actions)
        self.critic = nn.Linear(in_dim, 1)
        self.beta_v = beta_v
        self.beta_e = beta_e
        self.grad_eps = grad_eps #used in training to clip the gradient to avoid large gradients in backpropagation
        self.device = device

    def act(self, obs):
        with torch.no_grad():
            obs = obs.to(self.device).contiguous()
            dist, val = self.forward(obs)
            act = dist.sample()
            log_prob = dist.log_prob(act)

        return act.cpu(), val.cpu(), log_prob.cpu()

    def forward(self, obs):
        obs = self.model(obs)
        log = self.actor(obs)
        val = self.critic(obs).squeeze(1)
        dist = torch.distributions.Categorical(logits=log)
        return dist, val

    def policy_loss(self, log_pi, advantage):
        return log_pi * advantage

    def value_loss(self, value, future_reward):
        return self.beta_v * F.mse_loss(value, future_reward)

    def entropy_loss(self, dist):
        return self.beta_e * dist.entropy()
  
    def loss(self, batch, policy, value):
        b_obs, b_action, b_returns, b_delta, b_advantage, b_value = batch

        log_pi = policy.log_prob(b_action)

        pi_loss = self.policy_loss(log_pi, b_advantage)

        vf_loss = self.value_loss(value, b_returns)

        en_loss = self.entropy_loss(policy)

        # Return sum of loss (with appropriate sign)
        return torch.mean(-1 * (pi_loss - vf_loss + en_loss))
