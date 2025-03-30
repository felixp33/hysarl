import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def initialize_weights(module, gain=1):
    """Custom weight initialization"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.action_out = nn.Linear(hidden_dim, action_dim)
        self.action_dim = action_dim

        self.apply(lambda m: initialize_weights(m, gain=1.0))
        initialize_weights(self.action_out, gain=0.1)

    def forward(self, state):
        x = self.net(state)
        action = torch.tanh(self.action_out(x))  # Output in [-1, 1]
        return action


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


class TD3Agent:
    def __init__(self, state_dim, action_dim, replay_buffer, hidden_dim=256,
                 lr=3e-4, gamma=0.99, tau=0.005, grad_clip=1.0,
                 warmup_steps=5000, initial_noise_scale=0.1,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip

        # TD3 specific parameters
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        # Warm-up parameters
        self.warmup_steps = warmup_steps
        self.initial_noise_scale = initial_noise_scale
        self.total_steps = 0

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(
            state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1 = Critic(state_dim, action_dim,
                              hidden_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim,
                              hidden_dim).to(self.device)
        self.critic1_target = Critic(
            state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_target = Critic(
            state_dim, action_dim, hidden_dim).to(self.device)

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Setup optimizers with gradient clipping
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # Initialize trackers for monitoring
        self.train_iteration = 0
        self.actor_update_counter = 0
        self.critic_loss_history = []
        self.actor_loss_history = []
        self.td_error_history = []

    def get_config(self):
        """Return the configuration of the TD3 agent."""
        return {
            'hidden_dim': self.actor.net[0].out_features,
            'learning_rate': self.actor_optimizer.param_groups[0]['lr'],
            'gamma': self.gamma,
            'tau': self.tau,
            'policy_noise': self.policy_noise,
            'noise_clip': self.noise_clip,
            'policy_delay': self.policy_delay,
            'grad_clip': self.grad_clip,
            'warmup_steps': self.warmup_steps,
            'initial_noise_scale': self.initial_noise_scale,
            'device': str(self.device)
        }

    def select_action(self, state, evaluate=False):
        """Select action with warm-up exploration and proper noise handling."""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if evaluate:
                # Use deterministic action for evaluation
                action = self.actor(state)
            else:
                if self.total_steps < self.warmup_steps:
                    # Warmup phase: Sample purely random actions within [-1,1]
                    action = torch.tanh(
                        torch.normal(
                            mean=torch.zeros(
                                1, self.actor.action_dim).to(self.device),
                            std=self.initial_noise_scale
                        )
                    )
                else:
                    # Add exploration noise to deterministic action
                    action = self.actor(state)
                    noise = torch.normal(
                        mean=0,
                        std=self.initial_noise_scale *
                        (1.0 - min(1.0, self.total_steps / (2 * self.warmup_steps))),
                        size=action.shape
                    ).to(self.device)
                    action = (action + noise).clamp(-1, 1)

            return action.cpu().numpy()[0]

    def train(self, batch_size):
        """Performs a single training step of the TD3 algorithm."""

        if len(self.replay_buffer) < batch_size:
            return

        # Skip training during pure exploration phase
        if self.total_steps < self.warmup_steps // 2:
            return

        self.train_iteration += 1

        states, actions, rewards, next_states, dones, _ = self.replay_buffer.sample(
            batch_size)

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
            # Select next action and add target policy smoothing noise
            noise = torch.clamp(
                torch.randn_like(actions) * self.policy_noise,
                -self.noise_clip,
                self.noise_clip
            )

            next_actions = (self.actor_target(
                next_states) + noise).clamp(-1, 1)

            # Get target Q values
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)

            # Take minimum of Q values to mitigate overestimation bias
            q_next = torch.min(q1_next, q2_next)

            # Compute target Q value
            rewards_clamped = torch.clamp(rewards, -100, 100).view(-1, 1)
            target_q = rewards_clamped + (1 - dones) * self.gamma * q_next

        # Compute critic losses
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        # Track TD errors for diagnostics
        td_errors = torch.abs(
            target_q - current_q1).detach().cpu().numpy().flatten()
        td_errors_2 = torch.abs(
            target_q - current_q2).detach().cpu().numpy().flatten()
        self.td_error_history.append(
            0.5 * td_errors.mean() + 0.5 * td_errors_2.mean())

        # Compute and backpropagate critic losses
        critic1_loss = F.huber_loss(current_q1, target_q)
        critic2_loss = F.huber_loss(current_q2, target_q)

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

        # Update actor (delayed)
        actor_loss = torch.tensor(0.0)
        if self.train_iteration % self.policy_delay == 0:
            self.actor_update_counter += 1

            # Actor loss is negative of Q value
            actor_actions = self.actor(states)
            actor_loss = -self.critic1(states, actor_actions).mean()

            # Update actor with gradient clipping
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            # Update target networks
            self.safe_update_target(self.actor_target, self.actor)
            self.safe_update_target(self.critic1_target, self.critic1)
            self.safe_update_target(self.critic2_target, self.critic2)

            self.actor_loss_history.append(actor_loss.item())

        self.critic_loss_history.append(
            (critic1_loss.item() + critic2_loss.item()) / 2)
        self.total_steps += 1

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
            'critic_loss': np.mean(self.critic_loss_history[-100:]) if self.critic_loss_history else 0,
            'actor_loss': np.mean(self.actor_loss_history[-100:]) if self.actor_loss_history else 0,
            'td_error': np.mean(self.td_error_history[-100:]) if self.td_error_history else 0,
            'actor_updates': self.actor_update_counter,
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
                'actor_target_state_dict': self.actor_target.state_dict(),
                'critic1_state_dict': self.critic1.state_dict(),
                'critic2_state_dict': self.critic2.state_dict(),
                'critic1_target_state_dict': self.critic1_target.state_dict(),
                'critic2_target_state_dict': self.critic2_target.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
                'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
                'total_steps': self.total_steps,
                'train_iteration': self.train_iteration,
                'actor_update_counter': self.actor_update_counter,
                'td_error_history': self.td_error_history,
            }, path)
        except Exception as e:
            print(f"Error saving model: {e}")

    def load(self, path):
        """Load model with error handling"""
        try:
            checkpoint = torch.load(path)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.actor_target.load_state_dict(
                checkpoint['actor_target_state_dict'])
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
            self.total_steps = checkpoint.get('total_steps', 0)
            self.train_iteration = checkpoint.get('train_iteration', 0)
            self.actor_update_counter = checkpoint.get(
                'actor_update_counter', 0)
            if 'td_error_history' in checkpoint:
                self.td_error_history = checkpoint['td_error_history']
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            raise  # Re-raise the exception after logging for proper error handling upstream
