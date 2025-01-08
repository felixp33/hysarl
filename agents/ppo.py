import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, batch_size=64, epochs=10):
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epochs = epochs

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * \
                values[t + 1] * (1 - dones[t]) - values[t]
            last_advantage = delta + self.gamma * \
                (1 - dones[t]) * last_advantage
            advantages.insert(0, last_advantage)
        return advantages

    def train(self, memory):
        states, actions, log_probs, rewards, next_states, dones = memory

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Compute values and advantages
        values = self.critic(states).squeeze()
        next_values = self.critic(torch.FloatTensor(next_states)).squeeze()
        advantages = self.compute_advantages(rewards, values, dones)
        advantages = torch.FloatTensor(advantages)
        returns = advantages + values

        for _ in range(self.epochs):
            # Compute current log probabilities and values
            probs = self.actor(states)
            action_dist = torch.distributions.Categorical(probs)
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()

            # Ratio for importance sampling
            ratio = torch.exp(new_log_probs - log_probs)

            # Surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon,
                                1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

            # Critic loss
            critic_loss = nn.MSELoss()(values, returns)

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def store_experience(self, memory, state, action, log_prob, reward, next_state, done):
        memory[0].append(state)
        memory[1].append(action)
        memory[2].append(log_prob)
        memory[3].append(reward)
        memory[4].append(next_state)
        memory[5].append(done)

    def clear_memory(self, memory):
        for i in range(len(memory)):
            memory[i].clear()
