import numpy as np
import matplotlib.pyplot as plt
from debug_environment_parllel import EnvironmentOrchestrator
from dashboard import Dashboard
import time
from replay_buffer import ReplayBuffer  # if needed


def convert_env_id(env_id):
    """
    Given an environment id string like "gym_0", return the numeric index (0).
    """
    if isinstance(env_id, str) and '_' in env_id:
        try:
            return int(env_id.split('_')[1])
        except (IndexError, ValueError):
            pass
    return int(env_id)


class TrainingStats:
    def __init__(self, engines_dict):
        self.engines = []
        for engine, count in engines_dict.items():
            self.engines.extend([engine] * count)
        self.unique_engines = list(engines_dict.keys())
        self.engine_indices = {engine: [] for engine in self.unique_engines}
        for i, engine in enumerate(self.engines):
            self.engine_indices[engine].append(i)

        self.instance_rewards = {i: [] for i in range(len(self.engines))}
        self.type_rewards = {engine: [] for engine in self.unique_engines}

        # Track timing per instance instead of per engine type
        self.instance_start_times = {i: None for i in range(len(self.engines))}
        self.instance_step_times = {i: [] for i in range(len(self.engines))}
        self.episode_durations = {engine: [] for engine in self.unique_engines}

        # Add step tracking
        self.episode_steps = {engine: [] for engine in self.unique_engines}

    def start_instance_timing(self, instance_idx):
        """Start timing for a specific instance"""
        self.instance_start_times[instance_idx] = time.time()

    def end_instance_timing(self, instance_idx):
        """End timing for a specific instance and record the duration"""
        if self.instance_start_times[instance_idx] is not None:
            duration = time.time() - self.instance_start_times[instance_idx]
            self.instance_step_times[instance_idx].append(duration)
            self.instance_start_times[instance_idx] = None

    def compute_episode_durations(self):
        """Compute average episode duration for each engine type"""
        for engine_type in self.unique_engines:
            indices = self.engine_indices[engine_type]
            # Sum up all step times for instances of this engine type
            engine_durations = []
            for idx in indices:
                # If there are any times recorded
                if self.instance_step_times[idx]:
                    total_duration = sum(self.instance_step_times[idx])
                    engine_durations.append(total_duration)
                    # Clear the step times for next episode
                    self.instance_step_times[idx] = []

            if engine_durations:  # If we have any durations for this engine
                avg_duration = np.mean(engine_durations)
                self.episode_durations[engine_type].append(avg_duration)
            else:
                # Append the last duration or 0 if no history
                last_duration = self.episode_durations[engine_type][-1] if self.episode_durations[engine_type] else 0
                self.episode_durations[engine_type].append(last_duration)

    def update_rewards(self, episode_rewards):
        for i, reward in enumerate(episode_rewards):
            self.instance_rewards[i].append(reward)
        for engine_type in self.unique_engines:
            indices = self.engine_indices[engine_type]
            type_reward = np.mean([episode_rewards[i] for i in indices])
            self.type_rewards[engine_type].append(type_reward)

    def update_steps(self, env_steps):
        """Update steps for each engine type"""
        # Store raw steps per engine type without averaging
        for engine_type in self.unique_engines:
            indices = self.engine_indices[engine_type]
            # We want to store actual steps, not mean
            # or use any other logic that makes sense
            type_steps = max([env_steps[i] for i in indices])
            self.episode_steps[engine_type].append(type_steps)

    def get_stats(self):
        return {
            'instance': self.instance_rewards,
            'type': self.type_rewards,
            'episode_durations': self.episode_durations,
            'episode_steps': self.episode_steps
        }


class TrainingPipeline:
    def __init__(self, env_name, engines_dict, buffer_capacity, batch_size, episodes, steps_per_episode, agent):
        self.env_name = env_name
        self.engines_dict = engines_dict
        self.total_envs = sum(engines_dict.values())
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode

        self.envs = EnvironmentOrchestrator(env_name, engines_dict)
        self.rewards_history = []
        self.agent = agent
        self.stats = TrainingStats(engines_dict)

        # Add sample age tracking to dashboard parameters
        self.dashboard = Dashboard(
            self.total_envs,
            {'Environment': env_name,
             'Engines': engines_dict,
             'Buffer Capacity': buffer_capacity,
             'Batch Size': batch_size,
             'Episodes': episodes,
             'Steps per Episode': steps_per_episode}
        )

    def run(self):
        try:
            print("Entering TrainingPipeline.run() method")
            print("Number of environments:", self.total_envs)

            print("Checking agent:", self.agent)
            print("Checking replay buffer:", self.agent.replay_buffer)

            print("Attempting to reset environments")
            states = self.envs.reset()
            print("Successfully reset environments")

            plt.ion()

            for episode in range(self.episodes):
                print(f"Starting episode {episode + 1}/{self.episodes}")
                states = self.envs.reset()
                episode_rewards = [0 for _ in range(self.total_envs)]
                episode_dones = {i: False for i in range(self.total_envs)}
                active_envs = [True] * self.total_envs
                env_steps = [0] * self.total_envs
                global_step = 0

                while any(steps < self.steps_per_episode and active
                          for steps, active in zip(env_steps, active_envs)):
                    # Start timing for active environments
                    for env_idx, active in enumerate(active_envs):
                        if active and env_steps[env_idx] < self.steps_per_episode:
                            self.stats.start_instance_timing(env_idx)

                    actions = [
                        self.agent.select_action(state)
                        if active and steps < self.steps_per_episode
                        else None
                        for state, active, steps in zip(states, active_envs, env_steps)
                    ]

                    next_states, rewards, dones, env_ids = self.envs.step(
                        actions)
                    env_indices = [convert_env_id(eid) for eid in env_ids]

                    # End timing for environments that just took a step
                    for env_idx in env_indices:
                        if active_envs[env_idx] and env_steps[env_idx] < self.steps_per_episode:
                            self.stats.end_instance_timing(env_idx)

                    # First, collect experiences for active environments
                    for i, env_idx in enumerate(env_indices):
                        if active_envs[env_idx] and env_steps[env_idx] < self.steps_per_episode:
                            # Store the transition in replay buffer
                            self.agent.replay_buffer.push(
                                states[env_idx],
                                actions[env_idx],
                                rewards[i],
                                next_states[i],
                                dones[i],
                                env_ids[i],
                                episode
                            )
                            self.agent.total_steps += 1
                            episode_rewards[env_idx] += rewards[i]
                            env_steps[env_idx] += 1

                            # If done, mark environment as inactive but DON'T reset yet
                            if dones[i]:
                                episode_dones[env_idx] = True
                                active_envs[env_idx] = False

                    # After processing all experiences, then handle resets
                    reset_indices = [i for i, active in enumerate(
                        active_envs) if not active]
                    if reset_indices:
                        # Only reset environments that need resetting
                        reset_states = self.envs.reset_specific(reset_indices)

                        # Update states for reset environments
                        for idx, state in zip(reset_indices, reset_states):
                            states[idx] = state
                            # Reactivate the environment
                            active_envs[idx] = True

                    for i, next_state in zip(env_indices, next_states):
                        if active_envs[i] and env_steps[i] < self.steps_per_episode:
                            states[i] = next_state

                    self.agent.train(self.batch_size)
                    global_step += 1

                # Compute average episode durations for each engine type
                self.stats.compute_episode_durations()

                # Update steps for each engine type
                self.stats.update_steps(env_steps)

                mean_reward = np.mean(episode_rewards)
                self.rewards_history.append(mean_reward)
                self.stats.update_rewards(episode_rewards)

                # Update dashboard
                self.dashboard.update(
                    self.rewards_history,
                    self.agent.replay_buffer,
                    episode,
                    episode_dones,
                    self.stats
                )

                if episode % 10 == 0:
                    print(f"Episode {episode + 1}/{self.episodes}")
                    stats_data = self.stats.get_stats()
                    for engine_type, rewards in stats_data['type'].items():
                        steps = stats_data['episode_steps'][engine_type]
                        avg_duration = np.mean(
                            self.stats.episode_durations[engine_type][-10:])
                        print(f"{engine_type}:")
                        print(f"  Mean Reward: {rewards[-1]:.3f}")
                        print(f"  Mean Steps: {steps[-1]:.1f}")
                        print(f"  Avg Duration: {avg_duration:.3f}s")

        except Exception as e:
            print(f"Error during training: {e}")
            raise
        finally:
            self.envs.close()
            self.dashboard.close()
