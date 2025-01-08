import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class SACAgent:
    def __init__(self, state_dim, action_dim, replay_buffer, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Ensure actions are bounded
        )

        # Critic Networks
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Target Critic Networks
        self.critic1_target = nn.Sequential(*[layer for layer in self.critic1])
        self.critic2_target = nn.Sequential(*[layer for layer in self.critic2])
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state)
        return action.detach().numpy()[0]

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
            next_actions = self.actor(next_states)
            target_q1 = self.critic1_target(
                torch.cat([next_states, next_actions], dim=1))
            target_q2 = self.critic2_target(
                torch.cat([next_states, next_actions], dim=1))
            target_q = torch.min(target_q1, target_q2) - \
                self.alpha * next_actions
            target_value = rewards + (1 - dones) * self.gamma * target_q

        # Update critics
        current_q1 = self.critic1(torch.cat([states, actions], dim=1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=1))

        critic1_loss = nn.MSELoss()(current_q1, target_value)
        critic2_loss = nn.MSELoss()(current_q2, target_value)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor
        new_actions = self.actor(states)
        actor_loss = -torch.min(
            self.critic1(torch.cat([states, new_actions], dim=1)),
            self.critic2(torch.cat([states, new_actions], dim=1))
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.critic1_target, self.critic1)
        self.soft_update(self.critic2_target, self.critic2)

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def store_experience(self, state, action, reward, next_state, done, env_id):
        self.replay_buffer.push(state, action, reward,
                                next_state, done, env_id)
