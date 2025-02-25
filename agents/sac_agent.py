import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


def initialize_weights(module, gain=1):
    """Custom weight initialization"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-5, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim

        # Use LayerNorm for better training stability
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Initialize with smaller weights
        self.apply(lambda m: initialize_weights(m, gain=1.0))
        initialize_weights(self.mean, gain=0.1)
        initialize_weights(self.log_std, gain=0.1)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)

        # Numerically stable log_std
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)

        # Sample with numerical stability
        normal = Normal(mean, std + 1e-6)
        x = normal.rsample()

        # Squash sample
        action = torch.tanh(x)

        # Compute log probability with improved stability
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(torch.clamp(1 - action.pow(2), min=1e-6))
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize with appropriate scaling
        self.apply(lambda m: initialize_weights(m, gain=1.0))
        initialize_weights(self.net[-1], gain=0.1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class SACAgent:
    def __init__(self, state_dim, action_dim, replay_buffer, hidden_dim=256,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 target_entropy=None, grad_clip=1.0,
                 warmup_steps=5000, initial_noise_scale=0.1):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip

        # Warm-up parameters
        self.warmup_steps = warmup_steps
        self.initial_noise_scale = initial_noise_scale
        self.total_steps = 0

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1 = Critic(state_dim, action_dim,
                              hidden_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim,
                              hidden_dim).to(self.device)
        self.critic1_target = Critic(
            state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_target = Critic(
            state_dim, action_dim, hidden_dim).to(self.device)

        # Initialize target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Setup optimizers with gradient clipping
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # Automatic entropy tuning
        self.target_entropy = -action_dim if target_entropy is None else target_entropy
        self.log_alpha = torch.tensor(
            [np.log(0.5)], requires_grad=True, device=self.device)

        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Initialize trackers for monitoring
        self.train_iteration = 0
        self.critic_loss_history = []
        self.actor_loss_history = []
        self.alpha_loss_history = []
        self.entropy_history = []

    def select_action(self, state, evaluate=False):
        """Select action with warm-up exploration and proper noise handling."""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if evaluate:
                # Use deterministic action (mean) for evaluation
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                if self.total_steps < self.warmup_steps:
                    # Warmup phase: Sample purely random actions within [-1,1]
                    action = torch.tanh(
                        torch.normal(
                            mean=torch.zeros_like(
                                state[:, :self.actor.action_dim]),
                            std=self.initial_noise_scale
                        )
                    )
                else:
                    # Regular SAC sampling
                    # Actor outputs tanh-squashed action
                    raw_action, _ = self.actor.sample(state)

                    # Apply noise BEFORE tanh (to avoid going out of bounds)
                    noise_scale = max(0.0, self.initial_noise_scale *
                                      (1.0 - self.total_steps / (2 * self.warmup_steps)))
                    if noise_scale > 0:
                        noise = torch.normal(
                            mean=0.0, std=noise_scale, size=raw_action.shape, device=self.device)
                        raw_action += noise

                    # Apply tanh again for final squashing
                    action = torch.tanh(raw_action)

            return action.cpu().numpy()[0]

    def train(self, batch_size):
        """Performs a single training step of the SAC algorithm."""

        if len(self.replay_buffer) < batch_size:
            return

        # Skip training during pure exploration phase
        if self.total_steps < self.warmup_steps // 2:
            return

        self.train_iteration += 1

        # Compute current alpha
        self.alpha = torch.exp(self.log_alpha.detach())
        if self.total_steps < self.warmup_steps:
            self.alpha = torch.max(
                self.alpha, torch.tensor(0.2).to(self.device))

        # Sample from replay buffer
        states, actions, rewards, next_states, dones, _ = self.replay_buffer.sample(
            batch_size)

        # ✅ Fix float64-to-float32 conversion
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32,
                               device=self.device).view(-1, 1)
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32,
                             device=self.device).view(-1, 1)

        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)

            # ✅ Ensure correct shapes by taking `min()` without extra dimensions
            q_next = torch.min(q1_next, q2_next).view(-1, 1)

            # ✅ Clamp rewards and fix shape mismatch
            rewards_clamped = torch.clamp(rewards, -100, 100).view(-1, 1)
            target_q = rewards_clamped + \
                (1 - dones) * self.gamma * (q_next -
                                            self.alpha * next_log_probs.view(-1, 1))

        # ✅ Ensure `target_q` has correct shape `[batch_size, 1]`
        target_q = target_q.view(batch_size, 1).to(torch.float32)

        # Compute critic losses
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        # ✅ Fix shape mismatch for critic loss
        assert current_q1.shape == target_q.shape, f"Shape mismatch: {current_q1.shape} vs {target_q.shape}"
        assert current_q2.shape == target_q.shape, f"Shape mismatch: {current_q2.shape} vs {target_q.shape}"

        critic1_loss = F.huber_loss(current_q1, target_q)
        critic2_loss = F.huber_loss(current_q2, target_q)

        # Update critics with gradient clipping
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic1.parameters(), self.grad_clip)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic2.parameters(), self.grad_clip)
        self.critic2_optimizer.step()

        # Actor update
        new_actions, log_probs = self.actor.sample(states)
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_probs - q).mean()

        # Update actor with gradient clipping
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # Temperature (Alpha) update
        alpha_loss = -(self.log_alpha * (log_probs +
                       self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ✅ Ensure total_steps increments correctly
        self.total_steps += 1

        # Update target networks
        self.safe_update_target(self.critic1_target, self.critic1)
        self.safe_update_target(self.critic2_target, self.critic2)

        # Store training metrics
        self.critic_loss_history.append(
            (critic1_loss.item() + critic2_loss.item()) / 2)
        self.actor_loss_history.append(actor_loss.item())
        self.alpha_loss_history.append(alpha_loss.item())
        self.entropy_history.append(-log_probs.mean().item())

    def safe_update_target(self, target, source):
        """Safely update target network with error checking"""
        try:
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                if not (torch.isnan(source_param.data).any() or torch.isinf(source_param.data).any()):
                    target_param.data.copy_(
                        self.tau * source_param.data +
                        (1.0 - self.tau) * target_param.data
                    )
        except Exception as e:
            print(f"Error in target network update: {e}")

    def get_diagnostics(self):
        """Return training diagnostics"""
        return {
            'critic_loss': np.mean(self.critic_loss_history[-100:]),
            'actor_loss': np.mean(self.actor_loss_history[-100:]),
            'alpha_loss': np.mean(self.alpha_loss_history[-100:]),
            'entropy': np.mean(self.entropy_history[-100:]),
            'alpha': self.alpha.item(),
            'total_steps': self.total_steps,
            'exploration_status': self.get_exploration_status()
        }

    def get_exploration_status(self):
        """Return current exploration status"""
        if self.total_steps < self.warmup_steps:
            noise_scale = self.initial_noise_scale
            phase = "pure exploration"
        else:
            noise_scale = max(0.0,
                              self.initial_noise_scale *
                              (1.0 - self.total_steps / (2 * self.warmup_steps))
                              )
            phase = "policy-based exploration"

        return {
            'warmup_steps': self.warmup_steps,
            'current_noise_scale': noise_scale,
            'exploration_phase': phase
        }

    def save(self, path):
        """Save model with error handling"""
        try:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic1_state_dict': self.critic1.state_dict(),
                'critic2_state_dict': self.critic2.state_dict(),
                'critic1_target_state_dict': self.critic1_target.state_dict(),
                'critic2_target_state_dict': self.critic2_target.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
                'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
                'log_alpha': self.log_alpha,
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
                'total_steps': self.total_steps,
                'train_iteration': self.train_iteration
            }, path)
        except Exception as e:
            print(f"Error saving model: {e}")

    def load(self, path):
        """Load model with error handling"""
        try:
            checkpoint = torch.load(path)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            self.critic1_target.load_state_dict(
                checkpoint['critic1_target_state_dict'])
            self.critic2_target.load_state_dict(
                checkpoint['critic2_target_state_dict'])
            self.actor_optimizer.load_state_dict(
                checkpoint['actor_optimizer_state_dict'])
            self.critic1_optimizer.load_state_dict(
                checkpoint['critic1_optimizer_state_dict'])
            self.critic2_optimizer.load_state_dict(
                checkpoint['critic2_optimizer_state_dict'])
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(
                checkpoint['alpha_optimizer_state_dict'])
            self.total_steps = checkpoint.get(
                'total_steps', 0)  # Backward compatibility
            self.train_iteration = checkpoint.get(
                'train_iteration', 0)  # Backward compatibility
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            raise  # Re-raise the exception after logging for proper error handling upstream
