import numpy as np
import matplotlib.pyplot as plt
from src.dashboard import Dashboard

from src.environment_orchestrator import EnvironmentOrchestrator
from src.sequentiell.stats import TrainingStats


class TrainingPipeline:
    def __init__(self, env_name, engines_dict, buffer_capacity, batch_size, episodes, steps_per_episode, agent, engine_dropout=None, drop_out_limit=None):
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
        self.engine_dropout = engine_dropout
        self.drop_out_limit = drop_out_limit

        # Setup dashboard
        self.dashboard = Dashboard(
            self.total_envs,
            {'Environment': env_name,
             'Engines': engines_dict,
             'Buffer Capacity': buffer_capacity,
             'Batch Size': batch_size,
             'Episodes': episodes,
             'Steps per Episode': steps_per_episode,
             'Agent': agent.get_config(),
             }
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
                train_iterations = int(np.mean(list(episode_steps.values())))
                print(f"  Training iterations: {train_iterations}")
                self.agent.td_error_history = []

                # Drop out engines with low rewards, reweight the sampling distribution
                if self.engine_dropout and episode > 10:
                    treshold = np.max(
                        self.rewards_history[:self.total_envs * 10]) * self.drop_out_limit

                    for cur in self.engines_dict.keys():
                        if np.mean(self.stats.type_rewards[engine][:10]) < treshold:
                            drop_out_dict = {
                                engine_name: self.engines_dict[engine_name] / (1 - self.engines_dict[cur]) for engine_name in self.engines_dict.keys()}
                            drop_out_dict[engine] = 0
                            self.agent.replay_buffer.drop_out(
                                drop_out_dict)

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
                self.stats.update_td_errors(self.agent.td_error_history)
                # Get agent diagnostics
                agent_diagnostics = self.agent.get_diagnostics()

                # Update stats with agent diagnostics
                stats_data = self.stats.get_stats()
                stats_data['agent_diagnostics'] = agent_diagnostics

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
            self.stats.export_to_csv()
            self.envs.close()
            self.dashboard.close()
