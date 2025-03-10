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


class DiscreteActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim

        # Similar architecture but outputs logits for discrete actions
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.logits = nn.Linear(hidden_dim, action_dim)

        # Initialize with smaller weights for better initial exploration
        self.apply(lambda m: initialize_weights(m, gain=1.0))
        initialize_weights(self.logits, gain=0.1)

    def forward(self, state):
        x = self.net(state)
        logits = self.logits(x)
        return logits

    def sample(self, state, epsilon=1e-6):
        logits = self.forward(state)

        # Gumbel-Softmax trick for differentiable discrete sampling
        # Add epsilon for numerical stability
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + epsilon)

        # Sample action using Gumbel-Softmax
        action_probs = F.gumbel_softmax(logits, tau=1.0, hard=True)

        # Get action index
        actions = torch.argmax(action_probs, dim=-1)

        # Calculate log probability of the sampled action
        log_prob = torch.sum(action_probs * log_probs, dim=-1, keepdim=True)

        return actions, log_prob, probs


class DiscreteCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Q-network for each action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # Output Q-value for each action
        )

        # Initialize with appropriate scaling
        self.apply(lambda m: initialize_weights(m, gain=1.0))
        initialize_weights(self.net[-1], gain=0.1)

    def forward(self, state):
        return self.net(state)


class DiscreteSACAgent:
    def __init__(self, state_dim, action_dim, replay_buffer, hidden_dim=256,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 target_entropy=None, grad_clip=1.0,
                 warmup_steps=5000):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.action_dim = action_dim

        # Warm-up parameters
        self.warmup_steps = warmup_steps
        self.total_steps = 0

        # Initialize networks
        self.actor = DiscreteActor(
            state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1 = DiscreteCritic(
            state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = DiscreteCritic(
            state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_target = DiscreteCritic(
            state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_target = DiscreteCritic(
            state_dim, action_dim, hidden_dim).to(self.device)

        # Initialize target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Setup optimizers with gradient clipping
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # Automatic entropy tuning
        self.target_entropy = - \
            np.log(1.0/action_dim) * \
            0.98 if target_entropy is None else target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Initialize trackers for monitoring
        self.train_iteration = 0
        self.critic_loss_history = []
        self.actor_loss_history = []
        self.alpha_loss_history = []
        self.entropy_history = []

    def select_action(self, state, evaluate=False):
        """Select discrete action with exploration"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if evaluate:
                logits = self.actor(state)
                action = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            else:
                if self.total_steps < self.warmup_steps:
                    # Random action during warm-up
                    action = torch.tensor([np.random.randint(self.action_dim)])
                else:
                    # Sample action using current policy
                    action, _, _ = self.actor.sample(state)

            return action.cpu().numpy()[0]

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        # Skip training during pure exploration phase
        if self.total_steps < self.warmup_steps // 2:
            return

        self.train_iteration += 1

        # Compute current alpha
        self.alpha = torch.exp(self.log_alpha.detach())

        # Sample from replay buffer
        states, actions, rewards, next_states, dones, _ = self.replay_buffer.sample(
            batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Update critics
        with torch.no_grad():
            _, next_log_probs, next_probs = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states)
            q2_next = self.critic2_target(next_states)
            q_next = torch.min(q1_next, q2_next)

            # Calculate expected Q value with entropy
            v_next = torch.sum(
                next_probs * (q_next - self.alpha * next_log_probs), dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.gamma * v_next

        # Critic loss
        current_q1 = self.critic1(states).gather(1, actions.unsqueeze(1))
        current_q2 = self.critic2(states).gather(1, actions.unsqueeze(1))
        critic1_loss = F.mse_loss(current_q1, target_q.detach())
        critic2_loss = F.mse_loss(current_q2, target_q.detach())

        # Update critics
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
        _, log_probs, probs = self.actor.sample(states)
        q1 = self.critic1(states)
        q2 = self.critic2(states)
        q = torch.min(q1, q2)

        # Calculate actor loss with entropy
        inside_term = self.alpha.detach() * log_probs - q
        actor_loss = torch.sum(probs * inside_term, dim=1).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # Temperature update
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        alpha_loss = -(self.log_alpha * (entropy +
                       self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update target networks
        self.safe_update_target(self.critic1_target, self.critic1)
        self.safe_update_target(self.critic2_target, self.critic2)

        # Store metrics
        self.critic_loss_history.append(
            (critic1_loss.item() + critic2_loss.item()) / 2)
        self.actor_loss_history.append(actor_loss.item())
        self.alpha_loss_history.append(alpha_loss.item())
        self.entropy_history.append(entropy.mean().item())

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

    def store_experience(self, state, action, reward, next_state, done, env_id):
        """Store experience and update step counter"""
        self.replay_buffer.push(state, action, reward,
                                next_state, done, env_id)
        self.total_steps += 1

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
            phase = "pure exploration (random actions)"
        else:
            phase = "policy-based exploration"

        return {
            'warmup_steps': self.warmup_steps,
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
            self.total_steps = checkpoint.get('total_steps', 0)
            self.train_iteration = checkpoint.get('train_iteration', 0)
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            raise
