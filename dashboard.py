import matplotlib.pyplot as plt
import numpy as np

# TODO: imolement time obersavtion


class Dashboard:
    def __init__(self, num_envs, params):
        # Initialize 2x2 grid layout
        self.fig, self.axes = plt.subplots(
            2, 2, figsize=(12, 10))  # 2 rows, 2 columns
        plt.ion()  # Enable interactive mode for live updates

        # Display parameters at the top
        self.fig.suptitle(self.format_params(params), fontsize=10)

        # Track buffer distribution history
        self.buffer_history = {env_id: [] for env_id in range(num_envs)}
        self.episodes = []

    def format_params(self, params):
        """
        Formats parameters into a readable string.
        """
        return '\n'.join([f"{key}: {value}" for key, value in params.items()])

    def plot_rewards(self, rewards_history):
        """
        Plots the reward progression.
        """
        ax = self.axes[0, 0]  # Top-left plot
        ax.cla()  # Clear the plot
        ax.plot(rewards_history, label='Average Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True)

    def plot_buffer_distribution(self, replay_buffer):
        """
        Plots the replay buffer distribution by environment ID with percentages.
        """
        # Get distribution
        env_id_counts = replay_buffer.get_env_id_distribution()
        env_ids = list(env_id_counts.keys())
        counts = list(env_id_counts.values())

        # Compute percentages
        total_samples = sum(counts)
        percentages = [(count / total_samples) * 100 for count in counts]

        # Create bar plot
        ax = self.axes[0, 1]  # Top-right plot
        ax.cla()  # Clear the plot
        bars = ax.bar(env_ids, percentages, tick_label=[
                      f"Env {eid}" for eid in env_ids])

        # Add percentage labels above each bar
        for bar, percent in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{percent:.1f}%',  # Show percentage
                    ha='center', va='bottom', fontsize=10)

        # Set labels and title
        ax.set_xlabel('Environment ID')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Replay Buffer Distribution')
        ax.grid(True)

    def plot_buffer_trend(self):
        """
        Plots the buffer distribution trend over episodes.
        """
        ax = self.axes[1, 0]  # Bottom-left plot
        ax.cla()  # Clear the plot
        for env_id, counts in self.buffer_history.items():
            ax.plot(self.episodes, counts, label=f'Env {env_id}')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Sample Count')
        ax.set_title('Buffer Distribution Over Time')
        ax.legend()
        ax.grid(True)

    def update(self, rewards_history, replay_buffer, episode):
        """
        Updates the dashboard plots.
        """
        # Update buffer history
        env_id_counts = replay_buffer.get_env_id_distribution()
        total_samples = sum(env_id_counts.values())
        for env_id in self.buffer_history:
            count = env_id_counts.get(env_id, 0)
            # Track percentage over time
            self.buffer_history[env_id].append((count / total_samples) * 100)
        self.episodes.append(episode)

        # Update plots
        self.plot_rewards(rewards_history)
        self.plot_buffer_distribution(replay_buffer)
        self.plot_buffer_trend()
        plt.pause(0.01)  # Pause to allow GUI updates

    def close(self):
        """
        Closes the dashboard.
        """
        plt.ioff()
        plt.show()
