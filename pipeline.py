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
        plt.ion()  # Enable interactive mode for live plotting

        for episode in range(self.episodes):
            states = self.envs.reset()
            episode_rewards = [0 for _ in range(len(self.engines))]

            for step in range(self.steps_per_episode):
                # Select actions for each environment
                actions = [self.agent.select_action(state) for state in states]

                # Take a step in each environment
                next_states, rewards, dones, env_ids = self.envs.step(actions)

                # Store experiences and update rewards
                for i in range(len(self.engines)):
                    self.agent.replay_buffer.push(
                        states[i], actions[i], rewards[i], next_states[i], dones[i], env_ids[i]
                    )
                    episode_rewards[i] += rewards[i]

                # Train agent
                self.agent.train(self.batch_size)

                # Update states
                states = next_states

                # Handle done environments
                if all(dones):
                    break

            # Track and plot rewards
            self.rewards_history.append(np.mean(episode_rewards))
            print(
                f"Episode {episode + 1}/{self.episodes}, Rewards: {episode_rewards}")

            # Update dashboard with rewards and replay buffer distribution
            self.dashboard.update(self.rewards_history,
                                  self.agent.replay_buffer, episode)

        # Finalize plots
        plt.ioff()
        plt.show()

        # Close environments
        self.envs.close()
        self.dashboard.close()
