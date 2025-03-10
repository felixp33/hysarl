import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            # Output Q-values for each discrete action
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.net(state)


class DQNAgentDisc:
    def __init__(self, state_dim, action_dim, replay_buffer,
                 lr=3e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, target_update_freq=1000, tau=0.005,
                 reward_scale=1.0, gradient_clip=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.reward_scale = reward_scale
        self.gradient_clip = gradient_clip

        # Initialize total_steps counter
        self.total_steps = 0

        # Networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer with learning rate scheduler
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5000, verbose=False
        )

        # Device setup
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network.to(self.device)
        self.target_network.to(self.device)

    def select_action(self, state):
        self.total_steps += 1

        if self.total_steps < 10000 or np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return torch.argmax(q_values, dim=1).item()

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        # Sample experiences
        states, actions, rewards, next_states, dones, env_ids = self.replay_buffer.sample(
            batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(
            1).to(self.device) * self.reward_scale
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values using Double DQN
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            best_actions = next_q_values.max(1)[1].unsqueeze(1)
            next_q = self.target_network(next_states).gather(1, best_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss using Huber loss for stability
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.gradient_clip)
        self.optimizer.step()

        # Update target network
        if self.total_steps % self.target_update_freq == 0:
            self.soft_update_target_network()

        # Update epsilon
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def soft_update_target_network(self):
        """Soft update model parameters"""
        for target_param, param in zip(self.target_network.parameters(),
                                       self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def get_stats(self):
        """Return current agent statistics"""
        return {
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }
