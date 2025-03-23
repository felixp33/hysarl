from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import random
from collections import deque


class CompositionReplayBuffer:
    def __init__(self,
                 capacity: int,
                 strategy: str = 'uniform',
                 recency_bias: float = 0.0,
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
        self.sampling_composition = sampling_composition or {}
        self.engine_counts = engine_counts if engine_counts is not None else {
            'gym': 1}
        self.engine_types = list(self.engine_counts.keys())

        # Determine buffer composition
        if buffer_composition is not None:
            self.buffer_composition = buffer_composition
        elif sampling_composition is not None:
            self.buffer_composition = sampling_composition
        else:
            total_engines = sum(self.engine_counts.values())
            self.buffer_composition = {
                engine: count/total_engines
                for engine, count in self.engine_counts.items()
            }

        # Normalize buffer composition to ensure it sums to 1
        total = sum(self.buffer_composition.values())
        self.buffer_composition = {k: v/total for k,
                                   v in self.buffer_composition.items()}

        # Initialize buffers with precise capacity allocation
        self._init_buffers()

        # Initialize tracking variables
        self.sampled_counts = {}
        self.composition_tolerance = 0.02  # 2% tolerance
        self.current_episode = 0
        self.position = 0
        self._total_samples = 0
        self._last_composition = None
        self.recency_bias = recency_bias

    def _init_buffers(self):
        """
        Initialize buffers with precise capacity allocation based on buffer composition.
        """

        self.engine_buffers = {}

        # Ensure all engine types are represented in the buffer composition
        for engine_type in self.engine_types:
            if engine_type not in self.buffer_composition:
                self.buffer_composition[engine_type] = 0.0

        # Normalize buffer composition to ensure it sums to 1
        total_comp = sum(self.buffer_composition.values())
        self.buffer_composition = {
            k: v / total_comp for k, v in self.buffer_composition.items()
        }

        # Allocate exact capacity for each engine type
        for engine_type, composition in self.buffer_composition.items():
            # Calculate exact capacity for this engine type
            engine_capacity = max(1, int(self.capacity * composition))

            # Create buffer with exact max length
            self.engine_buffers[engine_type] = deque(maxlen=engine_capacity)

    def dropout(self, new_composition: Optional[Dict[str, float]]):
        self.sampling_composition = new_composition

    def push(self, state, action, reward, next_state, done, env_id, current_episode):
        """
        Add experience with comprehensive debugging for engine type extraction.
        """
        self.current_episode = current_episode

        # Extract engine type with multiple fallback strategies
        if '_' in env_id:
            engine_type = env_id.split('_')[0]
        elif len(self.engine_buffers) == 1:
            engine_type = list(self.engine_buffers.keys())[0]
        else:

            # Force engine type based on known environments
            if any(env in env_id for env in ['mujoco', 'Mujoco']):
                engine_type = 'mujoco'
            elif any(env in env_id for env in ['brax', 'Brax']):
                engine_type = 'brax'
            else:
                return  # Early return if no valid engine type

        # Validate and correct engine type
        if engine_type not in self.engine_buffers:

            engine_type = list(self.engine_buffers.keys())[0]

        # Prepare experience with type conversion
        experience = (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.float32(reward),
            np.array(next_state, dtype=np.float32),
            np.float32(done),
            env_id,
            current_episode
        )

        # Target buffer for this engine type
        target_buffer = self.engine_buffers[engine_type]

        # Add experience to buffer
        if len(target_buffer) < target_buffer.maxlen:
            target_buffer.append(experience)
        else:
            # If buffer is full, replace oldest experience
            target_buffer.popleft()
            target_buffer.append(experience)

        # Reset cached composition
        self._last_composition = None

        # Optional: Periodic composition logging
        if current_episode % 50 == 0:
            current_comp = self.get_current_composition()

            for eng, comp in current_comp.items():
                print(
                    f"{eng}: {comp * 100:.2f}% (buffer size: {len(self.engine_buffers[eng])})")

    def sample(self, batch_size: int) -> Tuple:
        """
        Wrapper method to call the appropriate sampling strategy

        Args:
            batch_size (int): Number of samples to collect

        Returns:
            Tuple of sampled experiences
        """
        if self.strategy == 'uniform':
            return self.uniform_sampling(batch_size)
        elif self.strategy == 'stratified':
            return self.stratified_sampling(batch_size)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")

    def stratified_sampling(self, batch_size: int) -> Tuple:
        """
        Perform stratified sampling with precise proportion control.

        Args:
            batch_size (int): Number of samples to collect

        Returns:
            Tuple of sampled experiences
        """

        # Rest of the existing implementation remains the same
        samples = []

        # Calculate samples per engine type based on sampling composition
        for engine_type, proportion in self.sampling_composition.items():
            # Calculate target number of samples for this engine type

            n_samples = max(0, int(proportion * batch_size))
            # Ensure the engine type exists in buffers
            if engine_type not in self.engine_buffers:

                continue

            buffer = self.engine_buffers[engine_type]

            # Sample from this engine's buffer
            if buffer and n_samples > 0:
                buffer_samples = self._recency_biased_sample_from_buffer(
                    buffer, n_samples)
                samples.extend(buffer_samples)

        # Fill remaining batch size if needed
        remaining = batch_size - len(samples)
        if remaining > 0:
            # Collect all experiences from all buffers
            all_experiences = []
            for buffer in self.engine_buffers.values():
                all_experiences.extend(buffer)

            # Sample remaining experiences
            if all_experiences:
                additional_samples = random.sample(
                    all_experiences,
                    min(remaining, len(all_experiences))
                )
                samples.extend(additional_samples)

        # Ensure exact batch size
        if len(samples) > batch_size:
            samples = random.sample(samples, batch_size)
        elif len(samples) < batch_size:
            # If still short, repeat sampling from all experiences
            all_experiences = []
            for buffer in self.engine_buffers.values():
                all_experiences.extend(buffer)

            while len(samples) < batch_size and all_experiences:
                samples.append(random.choice(all_experiences))

        # Log sampling composition for debugging
        sampling_counts = {}
        for sample in samples:
            # Extract engine type from env_id
            engine_type = sample[5].split('_')[0]
            sampling_counts[engine_type] = sampling_counts.get(
                engine_type, 0) + 1

        # Update sampled counts
        self._update_sampled_counts(samples)

        return self._prepare_batch(samples)

    def _update_sampled_counts(self, batch: List[Tuple]) -> None:
        """
        Track sampling distribution aggregated by engine type.
        """
        # Reset sampled counts before updating
        self.sampled_counts = {}

        # Aggregate samples by engine type
        engine_samples = {}
        for exp in batch:
            env_id = exp[5]  # env_id is at index 5
            engine_type = env_id.split('_')[0]

            # Count samples for specific environment ID
            if env_id not in self.sampled_counts:
                self.sampled_counts[env_id] = 0
            self.sampled_counts[env_id] += 1

            # Aggregate samples by engine type
            if engine_type not in engine_samples:
                engine_samples[engine_type] = 0
            engine_samples[engine_type] += 1

    def _prepare_batch(self, batch: List[Tuple]) -> Tuple:
        """Convert batch to numpy arrays efficiently with correct data types"""
        states, actions, rewards, next_states, dones, env_ids = zip(
            *[(exp[0], exp[1], exp[2], exp[3], exp[4], exp[5]) for exp in batch])

        return (
            np.array(states, dtype=np.float32),  # ✅ Ensure float32
            np.array(actions, dtype=np.float32),  # ✅ Ensure float32
            # ✅ Ensure float32
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),  # ✅ Ensure float32
            # ✅ Convert bool to float32
            np.array(dones, dtype=np.float32).reshape(-1, 1),
            np.array(env_ids)  # Keep as-is (likely string)
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
        """
        Calculate current composition with comprehensive debugging.
        """

        # Calculate current composition with caching
        if self._last_composition is None:
            total_experiences = sum(len(buffer)
                                    for buffer in self.engine_buffers.values())

            if total_experiences == 0:
                self._last_composition = {
                    engine: 0.0 for engine in self.engine_types
                }
            else:
                # Detailed composition calculation
                composition = {}
                for engine, buffer in self.engine_buffers.items():
                    buffer_size = len(buffer)
                    percentage = buffer_size / total_experiences
                    composition[engine] = percentage

                self._last_composition = composition

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
        """Get comprehensive buffer statistics including correct sample ages."""
        if not any(len(buffer) > 0 for buffer in self.engine_buffers.values()):
            return {
                "size": 0,
                "capacity": self.capacity,
                "fullness": 0.0,
                "composition": self.get_current_composition(),
                "sampling_distribution": self.sampled_counts,
                "sample_ages": {engine: 0 for engine in self.engine_types}
            }

        # Calculate average age per engine type
        average_ages = {}
        all_rewards = []

        for engine_type, buffer in self.engine_buffers.items():
            if len(buffer) > 0:
                experiences = list(buffer)
                rewards = [exp[2] for exp in experiences]
                all_rewards.extend(rewards)

                # Get episode numbers and calculate ages
                episode_numbers = [exp[6] for exp in experiences]
                if episode_numbers:
                    mean_episode = float(np.mean(episode_numbers))
                    age = self.current_episode - mean_episode
                    average_ages[engine_type] = age
                else:
                    average_ages[engine_type] = 0.0
            else:
                average_ages[engine_type] = 0.0

        return {
            "size": len(self),
            "capacity": self.capacity,
            "fullness": len(self) / self.capacity,
            "composition": self.get_current_composition(),
            "sampling_distribution": self.sampled_counts,
            "sample_ages": average_ages,
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

    def _recency_biased_sample_from_buffer(self, buffer, sample_size):
        """Sample with bias toward recent experiences from a specific buffer"""
        if not buffer or sample_size <= 0:
            return []

        buffer_list = list(buffer)
        buffer_size = len(buffer_list)

        if self.recency_bias <= 0.001 or buffer_size <= sample_size:
            # No recency bias or small buffer - use simple random sampling
            return random.sample(buffer_list, min(sample_size, buffer_size))

        # Generate indices with a skewed distribution favoring recent entries
        indices = []
        for _ in range(sample_size):
            # Use recency_bias as the alpha parameter for the power distribution
            r = np.random.random()
            skewed_value = r ** self.recency_bias

            # Map to an index, with recent items having higher probability
            idx = int(skewed_value * buffer_size)
            indices.append(buffer_size - 1 - idx)  # Reverse to get most recent

        # Get samples (handle index bounds)
        indices = [max(0, min(i, buffer_size-1)) for i in indices]
        return [buffer_list[i] for i in indices]
