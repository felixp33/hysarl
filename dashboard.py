import matplotlib.pyplot as plt
import numpy as np


class Dashboard:
    def __init__(self, num_envs, params):
        # Initialize 2x2 grid layout
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.ion()  # Enable interactive mode for live updates

        # Display parameters at the top
        self.fig.suptitle(self.format_params(params), fontsize=10)

        # Track buffer distribution history
        self.buffer_history = {env_id: [] for env_id in range(num_envs)}
        # Track done history for each environment
        self.done_history = {env_id: [] for env_id in range(num_envs)}
        self.episodes = []

        # Initialize moving average window size
        self.window_size = 10

    def format_params(self, params):
        """Formats parameters into a readable string."""
        return '\n'.join([f"{key}: {value}" for key, value in params.items()])

    def calculate_moving_average(self, data, window_size=10):
        """Calculates the moving average of the rewards."""
        if len(data) < window_size:
            return np.array(data)
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')

    def plot_rewards(self, rewards_history):
        """Plots the reward progression and moving average."""
        ax = self.axes[0, 0]  # Top-left plot
        ax.cla()  # Clear the plot

        # Plot raw rewards
        ax.plot(rewards_history, label='Raw Reward', alpha=0.3, color='gray')

        # Calculate and plot moving average
        if len(rewards_history) >= self.window_size:
            moving_avg = self.calculate_moving_average(
                rewards_history, self.window_size)
            # Adjust x-axis to align moving average with correct episodes
            ma_episodes = np.arange(self.window_size-1, len(rewards_history))
            ax.plot(ma_episodes, moving_avg,
                    label=f'{self.window_size}-Episode Moving Average',
                    color='blue', linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True)

    def plot_buffer_distribution(self, replay_buffer):
        """Plots the replay buffer distribution by environment ID with percentages."""
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
                    f'{percent:.1f}%',
                    ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Environment ID')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Replay Buffer Distribution')
        ax.grid(True)

    def plot_done_ratio(self):
        """Plots the ratio of dones per episode per environment."""
        ax = self.axes[1, 1]  # Bottom-right plot
        ax.cla()  # Clear the plot

        for env_id, dones in self.done_history.items():
            if dones:  # Only plot if we have data
                # Calculate cumulative dones ratio over episodes
                cumulative_dones = np.cumsum(dones)
                # Avoid division by zero
                episodes_array = np.array(self.episodes) + 1
                done_ratio = cumulative_dones / episodes_array

                ax.plot(self.episodes, done_ratio, label=f'Env {env_id}')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate (dones/episode)')
        ax.set_title('Environment Success Rate Over Time')
        ax.legend()
        ax.grid(True)

        # Set y-axis limits between 0 and 1 as it's a ratio
        ax.set_ylim([0, 1])

    def plot_buffer_trend(self):
        """Plots the buffer distribution trend over episodes."""
        ax = self.axes[1, 0]  # Bottom-left plot
        ax.cla()  # Clear the plot

        for env_id, counts in self.buffer_history.items():
            if counts:  # Only plot if we have data
                ax.plot(self.episodes, counts, label=f'Env {env_id}')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Buffer Sample Percentage (%)')
        ax.set_title('Buffer Distribution Over Time')
        ax.legend()
        ax.grid(True)

    def update(self, rewards_history, replay_buffer, episode, dones=None):
        """
        Updates the dashboard plots.
        Parameters:
            rewards_history: List of rewards over episodes
            replay_buffer: The replay buffer object
            episode: Current episode number
            dones: Dictionary mapping env_id to whether it was done this episode
        """
        # Update buffer history
        env_id_counts = replay_buffer.get_env_id_distribution()
        total_samples = sum(env_id_counts.values())
        for env_id in self.buffer_history:
            count = env_id_counts.get(env_id, 0)
            self.buffer_history[env_id].append((count / total_samples) * 100)

        # Update done history
        if dones is not None:
            for env_id in self.done_history:
                self.done_history[env_id].append(
                    1 if dones.get(env_id, False) else 0)

        self.episodes.append(episode)

        # Update all plots
        self.plot_rewards(rewards_history)
        self.plot_buffer_distribution(replay_buffer)
        self.plot_buffer_trend()
        self.plot_done_ratio()

        # Adjust layout and display
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.pause(0.01)  # Pause to allow GUI updates

    def close(self):
        """Closes the dashboard."""
        plt.ioff()
        plt.show()
