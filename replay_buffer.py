import numpy as np
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity, strategy='uniform'):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.strategy = strategy

    def push(self, state, action, reward, next_state, done, env_id):
        # Add experience to the buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(
                (state, action, reward, next_state, done, env_id))
        else:
            self.buffer[self.position] = (
                state, action, reward, next_state, done, env_id)
        self.position = (self.position + 1) % self.capacity  # Cyclic buffer

    def sample(self, batch_size, composition=None):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)  # Prevent sampling errors

        if self.strategy == 'uniform':
            return self.uniform_sampling(batch_size)
        elif self.strategy == 'stratified':
            if composition is None:
                raise ValueError(
                    "Stratified sampling requires a composition dictionary.")
            return self.stratified_sampling(batch_size, composition)

    def uniform_sampling(self, batch_size):
        # Uniform sampling
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, env_ids = zip(*batch)
        return states, actions, rewards, next_states, dones, env_ids

    def stratified_sampling(self, batch_size, composition):
        """
        Perform stratified sampling based on the proportions defined in the composition dictionary.
        :param batch_size: Total number of samples to draw.
        :param composition: Dictionary with env_id as key and proportion as value.
        :return: Sampled batch of experiences.
        """
        samples = []

        # Count total proportions
        total_proportions = sum(composition.values())

        # Group experiences by environment ID
        grouped = {}
        for experience in self.buffer:
            env_id = experience[-1]  # Last element is env_id
            if env_id not in grouped:
                grouped[env_id] = []
            grouped[env_id].append(experience)

        # Perform stratified sampling
        for env_id, proportion in composition.items():
            if env_id in grouped:
                # Calculate the number of samples to draw from this environment
                num_samples = int(
                    (proportion / total_proportions) * batch_size)
                samples.extend(random.sample(grouped[env_id], min(
                    num_samples, len(grouped[env_id]))))

        # Continue sampling from all available experiences to fill the remaining batch size
        remaining_samples = batch_size - len(samples)
        if remaining_samples > 0:
            additional_samples = random.choices(
                self.buffer, k=remaining_samples)  # Avoid sampling errors
            samples.extend(additional_samples)

        # Unpack the samples into separate arrays
        states, actions, rewards, next_states, dones, env_ids = zip(*samples)
        return states, actions, rewards, next_states, dones, env_ids

    def __len__(self):
        # Return current size of the buffer
        return len(self.buffer)

    def get_env_id_distribution(self):
        """
        Returns the count of samples grouped by env_id.
        """
        env_id_counts = {}
        for _, _, _, _, _, env_id in self.buffer:
            if env_id not in env_id_counts:
                env_id_counts[env_id] = 0
            env_id_counts[env_id] += 1
        return env_id_counts

    def clear(self):
        """
        Clears the replay buffer.
        """
        self.buffer = []
        self.position = 0
