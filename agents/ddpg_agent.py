import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class DDPGAgent:
    def __init__(self, state_dim, action_dim, replay_buffer, lr=1e-3, gamma=0.99, tau=0.005):
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.tau = tau

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Target Networks
        self.actor_target = nn.Sequential(*[layer for layer in self.actor])
        self.critic_target = nn.Sequential(*[layer for layer in self.critic])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        noise = noise_scale * np.random.randn(*action.shape)
        # Ensure actions are within bounds
        return np.clip(action + noise, -1.0, 1.0)

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones, _ = self.replay_buffer.sample(
            batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute target values
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(
                torch.cat([next_states, next_actions], dim=1))
            target_value = rewards + (1 - dones) * self.gamma * target_q

        # Update critic
        current_q = self.critic(torch.cat([states, actions], dim=1))
        critic_loss = nn.MSELoss()(current_q, target_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = - \
            self.critic(torch.cat([states, self.actor(states)], dim=1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def store_experience(self, state, action, reward, next_state, done, env_id):
        self.replay_buffer.push(state, action, reward,
                                next_state, done, env_id)
