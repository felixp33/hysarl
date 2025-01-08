import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class A3CNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A3CNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.policy = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )

        self.value = nn.Sequential(
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value


class A3CAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, entropy_beta=0.01):
        self.gamma = gamma
        self.entropy_beta = entropy_beta

        # Actor-Critic Network
        self.model = A3CNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        policy, _ = self.model(state)
        dist = Categorical(policy)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def compute_loss(self, rewards, log_probs, values, dones, entropies):
        Qvals = []
        Qval = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            Qval = reward + self.gamma * Qval * (1 - done)
            Qvals.insert(0, Qval)
        Qvals = torch.FloatTensor(Qvals)

        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        advantage = Qvals - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = F.mse_loss(values, Qvals.detach())
        entropy_loss = -entropies.mean()

        return actor_loss + 0.5 * critic_loss + self.entropy_beta * entropy_loss

    def train(self, experiences):
        states, actions, rewards, next_states, dones, log_probs, entropies = zip(
            *experiences)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)

        # Get values for states and next states
        _, values = self.model(states)
        _, next_values = self.model(next_states)
        values = values.squeeze()
        next_values = next_values.squeeze()

        # Compute loss and optimize
        loss = self.compute_loss(rewards, log_probs, values, dones, entropies)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_experience(self, state, action, reward, next_state, done, log_prob, entropy, experiences):
        experiences.append(
            (state, action, reward, next_state, done, log_prob, entropy))
