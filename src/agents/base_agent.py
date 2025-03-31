from typing import Dict, Optional
import numpy as np


class BaseAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 replay_buffer):
        """
        Base agent class for reinforcement learning algorithms.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            replay_buffer: Buffer for storing and sampling experiences
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.total_steps = 0

    def select_action(self, state, evaluate=False):
        """
        Select action based on current state.

        Args:
            state: Current state
            evaluate: Whether to use exploration or not

        Returns:
            Selected action
        """
        raise NotImplementedError("Subclasses must implement select_action")

    def train(self, batch_size):
        """
        Train agent on a batch of experiences.

        Args:
            batch_size: Number of samples to use for training
        """
        raise NotImplementedError("Subclasses must implement train")

    def get_diagnostics(self) -> Dict:
        """
        Return training diagnostics.

        Returns:
            Dictionary of diagnostic information
        """
        raise NotImplementedError("Subclasses must implement get_diagnostics")

    def get_config(self) -> Dict:
        """
        Return the configuration of the agent.

        Returns:
            Dictionary of configuration parameters
        """
        raise NotImplementedError("Subclasses must implement get_config")

    def save(self, path):
        """
        Save model with error handling.

        Args:
            path: Path to save the model
        """
        raise NotImplementedError("Subclasses must implement save")

    def load(self, path):
        """
        Load model with error handling.

        Args:
            path: Path to load the model from
        """
        raise NotImplementedError("Subclasses must implement load")
