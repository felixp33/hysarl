import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class TD3Agent:
    def __init__(self, state_dim, action_dim, replay_buffer, lr=1e-3, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.update_counter = 0

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
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

        # Target Networks
        self.actor_target = nn.Sequential(*[layer for layer in self.actor])
        self.critic1_target = nn.Sequential(*[layer for layer in self.critic1])
        self.critic2_target = nn.Sequential(*[layer for layer in self.critic2])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        noise = noise_scale * np.random.randn(*action.shape)
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

        # Select actions with noise for target policy smoothing
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            noise = (torch.randn_like(next_actions) *
                     self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)

            # Compute target Q-values
            target_q1 = self.critic1_target(
                torch.cat([next_states, next_actions], dim=1))
            target_q2 = self.critic2_target(
                torch.cat([next_states, next_actions], dim=1))
            target_q = torch.min(target_q1, target_q2)
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

        # Delay actor updates
        self.update_counter += 1
        if self.update_counter % self.policy_delay == 0:
            actor_loss = - \
                self.critic1(
                    torch.cat([states, self.actor(states)], dim=1)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic1_target, self.critic1)
            self.soft_update(self.critic2_target, self.critic2)

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def store_experience(self, state, action, reward, next_state, done, env_id):
        self.replay_buffer.push(state, action, reward,
                                next_state, done, env_id)
