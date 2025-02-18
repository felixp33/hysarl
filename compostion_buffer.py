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
        Initialize buffer with strict composition enforcement

        Args:
            capacity: Total buffer capacity
            strategy: Sampling strategy ('uniform' or 'stratified')
            sampling_composition: Target percentages for stratified sampling
            buffer_composition: Target percentages for buffer storage
            engine_counts: Dictionary mapping engine types to count
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
        self.engine_counts = engine_counts if engine_counts is not None else {
            'gym': 1}
        self.engine_types = list(self.engine_counts.keys())

        # Use provided buffer composition, DO NOT default to engine counts
        if buffer_composition is not None:
            self.buffer_composition = buffer_composition
        else:
            if sampling_composition is not None:
                # Use sampling composition as buffer composition if provided
                self.buffer_composition = sampling_composition
            else:
                # Only use engine counts as last resort
                total_engines = sum(self.engine_counts.values())
                self.buffer_composition = {
                    engine: count/total_engines
                    for engine, count in self.engine_counts.items()
                }

        # Normalize buffer composition to ensure it sums to 1
        total = sum(self.buffer_composition.values())
        self.buffer_composition = {
            k: v/total for k, v in self.buffer_composition.items()
        }

        print(f"Target buffer composition: {self.buffer_composition}")

        # Initialize buffers with exact capacities
        self._init_buffers()

        self.sampled_counts = {}
        self.composition_tolerance = 0.02  # 2% tolerance
        self.current_episode = 0
        self.position = 0
        self._total_samples = 0
        self._last_composition = None

    def _init_buffers(self):
        print("Starting buffer initialization...")
        print(f"Engine types: {self.engine_types}")
        print(f"Buffer composition: {self.buffer_composition}")

        self.engine_buffers = {}
        remaining_capacity = self.capacity

        # Make sure we have all engine types
        all_engines = set(self.engine_counts.keys()) | set(
            self.buffer_composition.keys())
        self.engine_types = list(all_engines)
        print(f"Combined engine types: {self.engine_types}")

        for engine_type in list(self.engine_types)[:-1]:
            print(f"Allocating capacity for {engine_type}")
            engine_capacity = int(
                self.capacity * self.buffer_composition[engine_type])
            engine_capacity = max(1, engine_capacity)
            self.engine_buffers[engine_type] = deque(maxlen=engine_capacity)
            remaining_capacity -= engine_capacity

        if self.engine_types:
            last_engine = self.engine_types[-1]
            print(f"Allocating remaining capacity for {last_engine}")
            self.engine_buffers[last_engine] = deque(
                maxlen=max(1, remaining_capacity))

        print(f"Final engine buffers: {list(self.engine_buffers.keys())}")

    def push(self, state, action, reward, next_state, done, env_id, current_episode):
        """Add experience with strict composition maintenance"""
        self.current_episode = current_episode
        engine_type = env_id.split('_')[0] if '_' in env_id else 'gym'

        if engine_type not in self.engine_buffers:
            raise ValueError(
                f"Unknown engine type: {engine_type}. Must be one of {list(self.buffer_composition.keys())}")

        target_buffer = self.engine_buffers[engine_type]
        experience = (state, action, reward, next_state, done, env_id)

        # Check composition before adding
        current_comp = self.get_current_composition()
        target_comp = self.buffer_composition[engine_type]

        if len(target_buffer) == target_buffer.maxlen:
            # If this engine is already at or above its target composition,
            # we need to make space by removing from over-represented engines
            if current_comp.get(engine_type, 0) >= target_comp:
                self.adjust_composition(engine_type)
            target_buffer.popleft()

        target_buffer.append(experience)
        self._last_composition = None  # Force composition recalculation

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences."""
        if batch_size > len(self):
            raise ValueError(
                f"Requested batch size {batch_size} is larger than buffer size {len(self)}")
        return self.uniform_sampling(batch_size) if self.strategy == 'uniform' else self.stratified_sampling(batch_size)

    def uniform_sampling(self, batch_size: int) -> Tuple:
        """Perform uniform sampling from all experiences."""
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
            all_experiences = []
            for buffer in self.engine_buffers.values():
                all_experiences.extend(buffer)

            if all_experiences:
                samples.extend(random.sample(all_experiences, remaining))

        self._update_sampled_counts(samples)
        return self._prepare_batch(samples)

    def _update_sampled_counts(self, batch: List[Tuple]) -> None:
        """Track sampling distribution."""
        for exp in batch:
            env_id = exp[5]  # env_id is at index 5
            self.sampled_counts[env_id] = self.sampled_counts.get(
                env_id, 0) + 1

    def _prepare_batch(self, batch: List[Tuple]) -> Tuple:
        """Convert batch to numpy arrays efficiently."""
        states, actions, rewards, next_states, dones, env_ids = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.bool_),
            np.array(env_ids)
        )

    def adjust_composition(self, target_engine: str) -> bool:
        """Strictly enforce composition by removing from over-represented engines"""
        current_comp = self.get_current_composition()

        # Calculate relative deviation from target for each engine
        deviations = {
            engine: (current_comp.get(
                engine, 0) - self.buffer_composition[engine]) / self.buffer_composition[engine]
            for engine in self.engine_types
        }

        # Find most over-represented engine
        over_represented = max(deviations.items(), key=lambda x: x[1])[0]

        if over_represented != target_engine and len(self.engine_buffers[over_represented]) > 0:
            # Remove oldest experience from most over-represented engine
            self.engine_buffers[over_represented].popleft()
            self._last_composition = None
            return True

        return False

    def get_current_composition(self) -> Dict[str, float]:
        """Calculate current composition with caching"""
        if self._last_composition is None:
            total_experiences = sum(len(buffer)
                                    for buffer in self.engine_buffers.values())
            if total_experiences == 0:
                self._last_composition = {
                    engine: 0.0 for engine in self.engine_types}
            else:
                self._last_composition = {
                    engine: len(buffer)/total_experiences
                    for engine, buffer in self.engine_buffers.items()
                }
        return self._last_composition

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
        self._total_samples = 0
        self._last_composition = None

    def __len__(self) -> int:
        """Get total number of experiences."""
        return sum(len(buffer) for buffer in self.engine_buffers.values())
