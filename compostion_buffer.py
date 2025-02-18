from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import random
from collections import deque


class CompositionReplayBuffer:
    def __init__(self,
                 capacity: int,
                 strategy: str = 'uniform',
                 sampling_composition: Optional[Dict[str, float]] = None,
                 buffer_composition: Optional[Dict[str, float]] = None,
                 engine_counts: Optional[Dict[str, int]] = None):
        """
        Initialize replay buffer with composition management and sampling options

        Args:
            capacity: Total buffer capacity
            strategy: Sampling strategy ('uniform' or 'stratified')
            sampling_composition: Target percentages for stratified sampling
            buffer_composition: Target percentages for buffer storage
            engine_counts: Dictionary mapping engine types to count (optional)
        """
        if strategy not in ['uniform', 'stratified']:
            raise ValueError(
                "Strategy must be either 'uniform' or 'stratified'")

        if strategy == 'stratified' and sampling_composition is None:
            raise ValueError(
                "Stratified sampling requires a sampling_composition")

        self.capacity = capacity
        self.strategy = strategy
        self.sampling_composition = sampling_composition

        # Handle the case where engine_counts is not provided initially
        self.engine_counts = engine_counts if engine_counts is not None else {
            'gym': 1}
        self.engine_types = list(self.engine_counts.keys())

        # Set buffer composition
        if buffer_composition is None:
            # Default to proportional by engine count
            total_engines = sum(self.engine_counts.values())
            self.buffer_composition = {
                engine: count/total_engines
                for engine, count in self.engine_counts.items()
            }
        else:
            # Validate and normalize buffer composition
            total = sum(buffer_composition.values())
            if not np.isclose(total, 1.0):
                raise ValueError("Buffer composition must sum to 1.0")
            self.buffer_composition = buffer_composition

        # Initialize buffers with capacities based on buffer composition
        self.engine_buffers = {}
        for engine_type in self.engine_types:
            engine_capacity = int(
                self.capacity * self.buffer_composition[engine_type])
            # Ensure minimum capacity of 1
            engine_capacity = max(1, engine_capacity)
            self.engine_buffers[engine_type] = deque(maxlen=engine_capacity)

        # Create mapping from env_id to engine type
        self.env_to_engine = {}
        for engine_type, count in self.engine_counts.items():
            for i in range(count):
                self.env_to_engine[f"{engine_type}_{i}"] = engine_type

        # For tracking sampling distribution and other stats
        self.sampled_counts: Dict[str, int] = {}
        self.composition_tolerance = 0.05  # 5% tolerance
        self.current_episode = 0
        self.position = 0

    def push(self,
             state: np.ndarray,
             action: Union[int, np.ndarray],
             reward: float,
             next_state: np.ndarray,
             done: bool,
             env_id: str,
             current_episode: int) -> None:
        """Add experience to the buffer."""
        self.current_episode = current_episode

        # Convert inputs to numpy arrays
        state = np.asarray(state)
        next_state = np.asarray(next_state)
        action = np.asarray(action)

        # Extract engine type from env_id
        engine_type = env_id.split('_')[0] if '_' in env_id else 'gym'

        # Ensure engine type exists in buffers
        if engine_type not in self.engine_buffers:
            self.engine_buffers[engine_type] = deque(
                maxlen=int(self.capacity * 0.5))
            self.engine_types.append(engine_type)
            # Default to equal split
            self.buffer_composition[engine_type] = 0.5

        experience = (state, action, reward, next_state,
                      done, env_id, current_episode)

        # Add to appropriate buffer
        target_buffer = self.engine_buffers[engine_type]
        if len(target_buffer) == target_buffer.maxlen:
            if self.composition_needs_adjustment(engine_type):
                if not self.adjust_composition(engine_type):
                    target_buffer.popleft()
            else:
                target_buffer.popleft()

        target_buffer.append(experience)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences."""
        if batch_size > len(self):
            raise ValueError(
                f"Requested batch size {batch_size} is larger than buffer size {len(self)}"
            )

        if self.strategy == 'uniform':
            return self.uniform_sampling(batch_size)
        else:  # stratified
            return self.stratified_sampling(batch_size)

    def uniform_sampling(self, batch_size: int) -> Tuple:
        """Perform uniform sampling from the buffer."""
        all_experiences = []
        for buffer in self.engine_buffers.values():
            all_experiences.extend(buffer)

        batch = random.sample(all_experiences, batch_size)
        self._update_sampled_counts(batch)
        return self._prepare_batch(batch)

    def stratified_sampling(self, batch_size: int) -> Tuple:
        """Sample according to specified composition."""
        samples = []

        # Sample from each group according to the desired composition
        for engine_type, proportion in self.sampling_composition.items():
            if engine_type not in self.engine_buffers:
                continue

            buffer = self.engine_buffers[engine_type]
            n_samples = int(proportion * batch_size)

            if n_samples > 0 and len(buffer) > 0:
                engine_samples = random.sample(
                    list(buffer),
                    min(n_samples, len(buffer))
                )
                samples.extend(engine_samples)

        # Fill remaining samples if needed
        remaining = batch_size - len(samples)
        if remaining > 0:
            available_experiences = []
            for buffer in self.engine_buffers.values():
                available_experiences.extend(buffer)

            # Remove already sampled experiences
            used_experiences = set(tuple(map(str, exp)) for exp in samples)
            available_experiences = [
                exp for exp in available_experiences
                if tuple(map(str, exp)) not in used_experiences
            ]

            if available_experiences:
                remaining_samples = random.sample(
                    available_experiences,
                    min(remaining, len(available_experiences))
                )
                samples.extend(remaining_samples)

        self._update_sampled_counts(samples)
        return self._prepare_batch(samples)

    def composition_needs_adjustment(self, engine_type: str) -> bool:
        """Check if adding to this engine would violate composition targets."""
        if len(self) == 0:
            return False

        current_comp = self.get_current_composition()
        target = self.buffer_composition[engine_type]
        current = current_comp[engine_type]
        return current > (target + self.composition_tolerance)

    def adjust_composition(self, target_engine: str) -> bool:
        """Remove experiences from over-represented buffers."""
        current_comp = self.get_current_composition()

        # Find most over-represented engine
        over_represented = None
        max_excess = -float('inf')

        for engine, target in self.buffer_composition.items():
            if engine == target_engine:
                continue
            excess = current_comp.get(engine, 0) - target
            if excess > max_excess and len(self.engine_buffers[engine]) > 0:
                max_excess = excess
                over_represented = engine

        if over_represented and max_excess > self.composition_tolerance:
            self.engine_buffers[over_represented].popleft()
            return True
        return False

    def _update_sampled_counts(self, batch: List[Tuple]) -> None:
        """Track sampling distribution."""
        for exp in batch:
            env_id = exp[5]  # env_id is at index 5
            self.sampled_counts[env_id] = self.sampled_counts.get(
                env_id, 0) + 1

    def _prepare_batch(self, batch: List[Tuple]) -> Tuple:
        """Convert list of experiences to batch of arrays."""
        states, actions, rewards, next_states, dones, env_ids, episodes = zip(
            *batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.bool_),
                np.array(env_ids), np.array(episodes))

    def get_current_composition(self) -> Dict[str, float]:
        """Calculate current composition of experiences."""
        total_experiences = sum(len(buffer)
                                for buffer in self.engine_buffers.values())
        if total_experiences == 0:
            return {engine: 0.0 for engine in self.engine_types}

        return {
            engine: len(buffer)/total_experiences
            for engine, buffer in self.engine_buffers.items()
        }

    def get_env_id_distribution(self) -> Dict[str, int]:
        """Get distribution of experiences across environment IDs."""
        env_id_counts = {}
        for buffer in self.engine_buffers.values():
            for exp in buffer:
                env_id = exp[5]  # env_id is at index 5
                env_id_counts[env_id] = env_id_counts.get(env_id, 0) + 1
        return env_id_counts

    def get_sampling_distribution(self) -> Dict[str, int]:
        """Get distribution of sampled experiences."""
        return self.sampled_counts

    def get_statistics(self) -> Dict[str, Union[float, int, Dict]]:
        """Get comprehensive buffer statistics."""
        if not any(len(buffer) > 0 for buffer in self.engine_buffers.values()):
            return {
                "size": 0,
                "capacity": self.capacity,
                "fullness": 0.0,
                "composition": self.get_current_composition(),
                "sampling_distribution": self.sampled_counts
            }

        all_rewards = []
        for buffer in self.engine_buffers.values():
            rewards = [exp[2] for exp in buffer]  # reward is at index 2
            all_rewards.extend(rewards)

        return {
            "size": len(self),
            "capacity": self.capacity,
            "fullness": len(self) / self.capacity,
            "composition": self.get_current_composition(),
            "sampling_distribution": self.sampled_counts,
            "reward_stats": {
                "mean": float(np.mean(all_rewards)) if all_rewards else 0.0,
                "std": float(np.std(all_rewards)) if all_rewards else 0.0,
                "min": float(np.min(all_rewards)) if all_rewards else 0.0,
                "max": float(np.max(all_rewards)) if all_rewards else 0.0
            },
            "unique_envs": len(self.get_env_id_distribution())
        }

    def clear(self) -> None:
        """Clear all experiences from all buffers."""
        for buffer in self.engine_buffers.values():
            buffer.clear()
        self.sampled_counts.clear()
        self.position = 0

    def __len__(self) -> int:
        """Return current size of the buffer."""
        return sum(len(buffer) for buffer in self.engine_buffers.values())
