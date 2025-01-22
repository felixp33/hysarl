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
            plt.ion()

            for episode in range(self.episodes):
                states = self.envs.reset()
                episode_rewards = [0 for _ in range(len(self.engines))]
                episode_dones = {i: False for i in range(len(self.engines))}
                active_envs = [True] * len(self.engines)
                step_count = 0

                while step_count < self.steps_per_episode:
                    # Select actions
                    actions = [
                        self.agent.select_action(state) if active
                        else None
                        for state, active in zip(states, active_envs)
                    ]

                    # Take step and handle potential empty results
                    try:
                        next_states, rewards, dones, env_ids = self.envs.step(
                            actions)
                    except ValueError as e:
                        if not any(active_envs):
                            break  # All environments are done
                        raise  # Re-raise if it's a different issue

                    # Store experiences and update states
                    for i, env_id in enumerate(env_ids):
                        if active_envs[env_id]:
                            self.agent.replay_buffer.push(
                                states[env_id],
                                actions[env_id],
                                rewards[i],
                                next_states[i],
                                dones[i],
                                env_id
                            )
                            episode_rewards[env_id] += rewards[i]

                            if dones[i]:
                                episode_dones[env_id] = True
                                active_envs[env_id] = False

                    # Update states only for active environments
                    for i, next_state in zip(env_ids, next_states):
                        if active_envs[i]:
                            states[i] = next_state

                    # Train agent
                    self.agent.train(self.batch_size)
                    step_count += 1

                    if not any(active_envs):
                        break

                # Update metrics
                mean_reward = np.mean(episode_rewards)
                self.rewards_history.append(mean_reward)

                if episode % 10 == 0:  # Log less frequently
                    print(f"Episode {episode + 1}/{self.episodes}, " +
                          f"Mean Reward: {mean_reward:.3f}, " +
                          f"Steps: {step_count}")

                self.dashboard.update(
                    self.rewards_history,
                    self.agent.replay_buffer,
                    episode,
                    episode_dones
                )

        except Exception as e:
            print(f"Error during training: {e}")
            raise
        finally:
            self.envs.close()
            self.dashboard.close()
