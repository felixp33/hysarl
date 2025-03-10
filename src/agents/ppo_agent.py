import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def push(self, state, action, log_prob, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

    def get_batch(self):
        return (torch.FloatTensor(self.states),
                torch.LongTensor(self.actions),
                torch.FloatTensor(self.log_probs),
                torch.FloatTensor(self.rewards),
                torch.FloatTensor(self.next_states),
                torch.FloatTensor(self.dones))


class ActionResult:
    """Helper class to prevent action/log_prob confusion"""

    def __init__(self, action, log_prob):
        self.action = action
        self.log_prob = log_prob


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2,
                 batch_size=64, epochs=10, clip_grad=0.5):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epochs = epochs
        self.clip_grad = clip_grad
        self.action_dim = action_dim  # Store action_dim for validation

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = PPOMemory()

    def get_action_distribution(self, state):
        action_probs = self.actor(state)
        return Categorical(action_probs)

    def select_action(self, state):
        """
        Returns an ActionResult object to prevent confusion between action and log_prob
        """
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist = self.get_action_distribution(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Ensure action is a valid discrete action
        action_value = action.item()
        if not (0 <= action_value < self.action_dim):
            raise ValueError(f"Invalid action value {action_value}")

        return ActionResult(action_value, log_prob.item())

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * \
                next_value * (1 - dones[t]) - values[t]
            last_advantage = delta + self.gamma * \
                (1 - dones[t]) * last_advantage
            advantages.insert(0, last_advantage)

        advantages = torch.tensor(advantages, device=self.device)
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)
        return advantages

    def train(self):
        """Train the agent using collected experiences"""
        if len(self.memory.states) == 0:
            return

        states, actions, old_log_probs, rewards, next_states, dones = self.memory.get_batch()
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        with torch.no_grad():
            values = self.critic(states).squeeze()
            advantages = self.compute_advantages(
                rewards, values.cpu().numpy(), dones)
            returns = advantages + values
            advantages = advantages.to(self.device)
            returns = returns.to(self.device)

        for _ in range(self.epochs):
            dist = self.get_action_distribution(states)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon,
                                1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

            current_values = self.critic(states).squeeze()
            values_clipped = values + \
                torch.clamp(current_values - values, -
                            self.epsilon, self.epsilon)
            critic_loss1 = (current_values - returns).pow(2)
            critic_loss2 = (values_clipped - returns).pow(2)
            critic_loss = 0.5 * torch.max(critic_loss1, critic_loss2).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.clip_grad)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.clip_grad)
            self.critic_optimizer.step()

        self.memory.clear()

    def store_experience(self, state, action_result, reward, next_state, done):
        """
        Store a transition in memory
        action_result should be an ActionResult object containing both action and log_prob
        """
        self.memory.push(
            state,
            action_result.action,
            action_result.log_prob,
            reward,
            next_state,
            done
        )
