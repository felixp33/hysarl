import numpy as np
import matplotlib.pyplot as plt
from src.dashboard import Dashboard
import time

from src.environment import EnvironmentOrchestrator


class TrainingStats:
    def __init__(self, engines_dict):
        self.engines_dict = engines_dict
        self.unique_engines = list(engines_dict.keys())

        # Track rewards per engine type
        self.type_rewards = {engine: [] for engine in self.unique_engines}

        # Track timing per engine type
        self.start_times = {engine: None for engine in self.unique_engines}
        self.step_times = {engine: [] for engine in self.unique_engines}
        self.episode_durations = {engine: [] for engine in self.unique_engines}

        # Track steps per engine type
        self.episode_steps = {engine: [] for engine in self.unique_engines}

        # Create index mapping for dashboard compatibility
        self.engine_indices = {}
        env_id = 0
        for engine, count in engines_dict.items():
            self.engine_indices[engine] = []
            for _ in range(count):
                self.engine_indices[engine].append(env_id)
                env_id += 1

        # Track rewards per instance (for dashboard compatibility)
        self.instance_rewards = {i: []
                                 for i in range(sum(engines_dict.values()))}

    def start_instance_timing(self, engine):
        """Start timing for a specific engine"""
        self.start_times[engine] = time.time()

    def end_instance_timing(self, engine):
        """End timing for a specific engine and record duration"""
        if self.start_times[engine] is not None:
            duration = time.time() - self.start_times[engine]
            self.step_times[engine].append(duration)
            self.start_times[engine] = None

    def compute_episode_durations(self):
        """Compute average episode duration for each engine type"""
        for engine in self.unique_engines:
            if self.step_times[engine]:
                # Sum up all step times for this engine
                total_duration = sum(self.step_times[engine])
                self.episode_durations[engine].append(total_duration)
                # Clear the step times for next episode
                self.step_times[engine] = []
            else:
                # If no steps were recorded, use previous duration or 0
                last_duration = self.episode_durations[engine][-1] if self.episode_durations[engine] else 0
                self.episode_durations[engine].append(last_duration)

    def update_rewards(self, rewards_dict):
        """Update rewards for each engine type"""
        for engine, reward in rewards_dict.items():
            self.type_rewards[engine].append(reward)

            # Update instance rewards for dashboard compatibility
            for idx in self.engine_indices[engine]:
                self.instance_rewards[idx].append(reward)

    def update_steps(self, steps_dict):
        """Update steps for each engine type"""
        for engine, steps in steps_dict.items():
            self.episode_steps[engine].append(steps)

    def get_stats(self):
        """Get all statistics in a format compatible with the dashboard"""
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

        # Setup dashboard
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
            print(
                f"Starting sequential training with {len(self.engines_dict)} engine types")
            plt.ion()  # Interactive plotting mode

            for episode in range(self.episodes):
                print(f"\nEpisode {episode + 1}/{self.episodes}")

                episode_rewards = {}
                episode_steps = {}
                episode_dones = {}

                # Run one episode for each engine type
                for engine_type in self.engines_dict.keys():
                    print(f"  Training on {engine_type}...")

                    # Run episode for this engine
                    reward, steps, done = self.envs.run_episode(
                        engine_type=engine_type,
                        agent=self.agent,
                        steps_per_episode=self.steps_per_episode,
                        episode_num=episode,
                        stats=self.stats
                    )

                    # Store results
                    episode_rewards[engine_type] = reward
                    episode_steps[engine_type] = steps

                    # For each instance of this engine (for dashboard compatibility)
                    for i in range(self.engines_dict[engine_type]):
                        idx = self.stats.engine_indices[engine_type][i]
                        episode_dones[idx] = done

                    print(
                        f"  {engine_type}: Reward = {reward:.2f}, Steps = {steps}")

                    # Train the agent multiple times after each environment episode
                    train_iterations = steps // 10  # Adjust this multiplier as needed
                    print(f"  Training iterations: {train_iterations}")
                    for _ in range(train_iterations):
                        if len(self.agent.replay_buffer) >= self.batch_size:
                            self.agent.train(self.batch_size)

                # Calculate average reward across all engines (weighted by count)
                total_weighted_reward = 0
                total_weight = 0
                for engine, reward in episode_rewards.items():
                    count = self.engines_dict[engine]
                    total_weighted_reward += reward * count
                    total_weight += count

                mean_reward = total_weighted_reward / total_weight
                self.rewards_history.append(mean_reward)

                # Update stats
                self.stats.update_rewards(episode_rewards)
                self.stats.update_steps(episode_steps)
                self.stats.compute_episode_durations()

                # Update dashboard
                self.dashboard.update(
                    self.rewards_history,
                    self.agent.replay_buffer,
                    episode,
                    episode_dones,
                    self.stats
                )

                # Print periodic summary
                if episode % 10 == 0 or episode == self.episodes - 1:
                    print(f"\nEpisode {episode + 1}/{self.episodes} Summary:")
                    stats_data = self.stats.get_stats()
                    for engine_type, rewards in stats_data['type'].items():
                        steps = stats_data['episode_steps'][engine_type]
                        avg_duration = np.mean(
                            self.stats.episode_durations[engine_type][-10:]) if self.stats.episode_durations[engine_type] else 0
                        print(f"{engine_type}:")
                        print(f"  Mean Reward: {rewards[-1]:.3f}")
                        print(f"  Mean Steps: {steps[-1]:.1f}")
                        print(f"  Avg Duration: {avg_duration:.3f}s")

            print("✅ Training complete!")

        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self.envs.close()
            self.dashboard.close()
