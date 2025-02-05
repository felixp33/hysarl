from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity: int, strategy: str = 'uniform', composition: Optional[Dict[str, float]] = None):
        if strategy not in ['uniform', 'stratified']:
            raise ValueError("Strategy must be in ['uniform', 'stratified']")

        if strategy == 'stratified' and composition is None:
            raise ValueError(
                "Stratified sampling requires a composition dictionary")

        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.position = 0
        self.strategy = strategy
        self.composition = composition

    def sample(self, batch_size: int) -> Tuple:
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Requested batch size {batch_size} is larger than buffer size {len(self.buffer)}")

        if self.strategy == 'uniform':
            return self.uniform_sampling(batch_size)
        return self.stratified_sampling(batch_size)

    def push(self,
             state: np.ndarray,
             action: Union[int, np.ndarray],
             reward: float,
             next_state: np.ndarray,
             done: bool,
             env_id: str) -> None:
        """Add an experience to the buffer.

        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether the episode ended
            env_id: Environment identifier for stratified sampling
        """
        # Convert inputs to numpy arrays if they aren't already
        state = np.asarray(state)
        next_state = np.asarray(next_state)
        action = np.asarray(action)

        experience = (state, action, reward, next_state, done, env_id)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self,
               batch_size: int,
               composition: Optional[Dict[str, float]] = None) -> Tuple:
        """Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample
            composition: Dictionary mapping env_ids to sampling proportions
                        Required for stratified sampling

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, env_ids)

        Raises:
            ValueError: If batch_size > buffer size or if composition is missing
                       for stratified sampling
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Requested batch size {batch_size} is larger than buffer size {len(self.buffer)}")

        if self.strategy == 'uniform':
            return self.uniform_sampling(batch_size)
        elif self.strategy == 'stratified':
            return self.stratified_sampling(batch_size)

    def uniform_sampling(self, batch_size: int) -> Tuple:
        """Perform uniform sampling from the buffer.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, env_ids)
        """
        batch = random.sample(self.buffer, batch_size)
        return self._prepare_batch(batch)

    def stratified_sampling(self, batch_size: int) -> Tuple:
        samples = []

        # Group experiences by engine type
        grouped = {}
        for exp in self.buffer:
            # Extract engine type from env_id
            engine_type = exp[-1].split('_')[0]
            grouped.setdefault(engine_type, []).append(exp)

        # Normalize proportions
        total = sum(self.composition.values())
        normalized_composition = {
            k: v/total for k, v in self.composition.items()}

        # Sample from each group according to composition
        for engine_type, proportion in normalized_composition.items():
            if engine_type in grouped:
                n_samples = int(proportion * batch_size)
                if n_samples > 0:
                    group_samples = random.sample(
                        grouped[engine_type],
                        min(n_samples, len(grouped[engine_type]))
                    )
                    samples.extend(group_samples)

        # Fill remaining samples using uniform sampling from unused experiences
        remaining = batch_size - len(samples)
        if remaining > 0:
            used_experiences = set(tuple(map(lambda x: x.tobytes() if isinstance(x, np.ndarray) else x, exp))
                                   for exp in samples)
            available_experiences = [
                exp for exp in self.buffer
                if tuple(map(lambda x: x.tobytes() if isinstance(x, np.ndarray) else x, exp)) not in used_experiences
            ]
            if available_experiences:
                remaining_samples = random.sample(
                    available_experiences,
                    min(remaining, len(available_experiences))
                )
                samples.extend(remaining_samples)

        return self._prepare_batch(samples)

    def decay_by_episode(self, decay_by_episode) -> Tuple:
        """varies sampling composition by episode
            - decrease sampling of low quality by fixed rate
            - decrease sampling of low quality to reach 100% by fixed episode
        """
        pass

    def progress_decay_avoidence(self, decay_by_episode) -> Tuple:
        """
        - decrease sampling in low quality if 
        """
        pass

    def _prepare_batch(self, batch: List[Tuple]) -> Tuple:
        """Convert a list of experiences into a batch of arrays.

        Args:
            batch: List of experience tuples

        Returns:
            Tuple of numpy arrays for each component
        """
        states, actions, rewards, next_states, dones, env_ids = zip(*batch)

        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.bool_)

        return states, actions, rewards, next_states, dones, env_ids

    def get_env_id_distribution(self) -> Dict[str, int]:
        """Get the distribution of experiences across environments.

        Returns:
            Dictionary mapping environment IDs to counts
        """
        env_id_counts = {}
        for exp in self.buffer:
            env_id = exp[-1]
            env_id_counts[env_id] = env_id_counts.get(env_id, 0) + 1
        return env_id_counts

    def get_statistics(self) -> Dict[str, Union[float, int]]:
        """Calculate basic statistics of the stored experiences.

        Returns:
            Dictionary containing buffer statistics
        """
        if not self.buffer:
            return {
                "size": 0,
                "capacity": self.capacity,
                "fullness": 0.0
            }

        rewards = [exp[2] for exp in self.buffer]

        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "fullness": len(self.buffer) / self.capacity,
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
            "unique_envs": len(self.get_env_id_distribution())
        }

    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer = []
        self.position = 0

    def __len__(self) -> int:
        """Get the current size of the buffer."""
        return len(self.buffer)
