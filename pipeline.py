import numpy as np
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer
from agents.dqn_agent import DQNAgent
from environment import EnvironmentOrchestrator


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

    def run(self):
        for episode in range(self.episodes):
            states = self.envs.reset()
            episode_rewards = [0 for _ in range(len(self.engines))]

            for step in range(self.steps_per_episode):
                # Select actions for each environment
                actions = [self.agent.select_action(state) for state in states]

                # Take a step in each environment
                next_states, rewards, dones = self.envs.step(actions)

                # Store experiences and update rewards
                for i in range(len(self.engines)):
                    self.agent.replay_buffer.push(
                        states[i], actions[i], rewards[i], next_states[i], dones[i])
                    episode_rewards[i] += rewards[i]

                # Train agent
                self.agent.train(self.batch_size)

                # Update states
                states = next_states

                # Handle done environments
                if all(dones):
                    break

            self.rewards_history.append(np.mean(episode_rewards))
            print(
                f"Episode {episode + 1}/{self.episodes}, Rewards: {episode_rewards}")

        # Close environments
        self.envs.close()

    def plot_rewards(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.rewards_history, label='Average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.show()
