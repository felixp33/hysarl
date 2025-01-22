import numpy as np
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer
from agents.dqn_agent import DQNAgent
from environment import EnvironmentOrchestrator

import numpy as np
import matplotlib.pyplot as plt
from environment import EnvironmentOrchestrator
from dashboard import Dashboard  # Import the dashboard


class TrainingPipeline:
    def __init__(self, env_name, engines, buffer_capacity, batch_size, episodes, steps_per_episode, agent):
        self.env_name = env_name
        self.engines = engines
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode

        # Initialize environment orchestrator
        self.envs = EnvironmentOrchestrator(env_name, engines)

        # Track rewards for visualization
        self.rewards_history = []

        # Set the agent
        self.agent = agent

        # Initialize the dashboard for visualization
        self.dashboard = Dashboard(
            len(engines), {'Environment: ': env_name, 'Engines: ': engines, 'Buffer Capacity: ': buffer_capacity, 'Batch Size: ': batch_size, 'Episodes: ': episodes, 'Steps per Episode: ': steps_per_episode})

    def run(self):
        try:
            plt.ion()  # Enable interactive plotting

            for episode in range(self.episodes):
                states = self.envs.reset()
                episode_rewards = [0 for _ in range(len(self.engines))]
                episode_dones = {i: False for i in range(
                    len(self.engines))}  # Track dones
                active_envs = [True] * len(self.engines)

                for step in range(self.steps_per_episode):
                    # Select actions only for active environments
                    actions = []
                    for i, state in enumerate(states):
                        if active_envs[i]:
                            actions.append(self.agent.select_action(state))
                        else:
                            actions.append(None)

                    # Take step in each environment
                    next_states, rewards, dones, env_ids = self.envs.step(
                        actions)

                    # Store experiences and update rewards only for active envs
                    for i in range(len(self.engines)):
                        if active_envs[i]:
                            self.agent.replay_buffer.push(
                                states[i], actions[i], rewards[i], next_states[i], dones[i], env_ids[i]
                            )
                            episode_rewards[i] += rewards[i]

                            # Update episode_dones and active_envs when environments finish
                            if dones[i]:
                                episode_dones[i] = True
                                active_envs[i] = False

                    # Train agent
                    self.agent.train(self.batch_size)

                    # Update states
                    states = next_states

                    # Break if all environments are done
                    if not any(active_envs):
                        break

                # Track and plot rewards
                mean_reward = np.mean(episode_rewards)
                self.rewards_history.append(mean_reward)
                print(
                    f"Episode {episode + 1}/{self.episodes}, Rewards: {episode_rewards}")

                # Update dashboard with all metrics
                self.dashboard.update(
                    self.rewards_history,
                    self.agent.replay_buffer,
                    episode,
                    episode_dones
                )

            # Finalize plots
            plt.ioff()
            plt.show()

        except Exception as e:
            print(f"Error during training: {e}")
            raise

        finally:
            # Ensure proper cleanup
            self.envs.close()
            self.dashboard.close()
