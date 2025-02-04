import numpy as np
import matplotlib.pyplot as plt


class Dashboard:
    def __init__(self, num_envs, params):
        # Initialize 2x3 grid layout
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        plt.ion()

        # Display parameters at the top
        self.fig.suptitle(self.format_params(params), fontsize=10)

        # Track histories
        self.done_history = {env_id: [] for env_id in range(num_envs)}
        self.samples_history = {env_id: [] for env_id in range(num_envs)}
        self.buffer_composition_history = {
            env_id: [] for env_id in range(num_envs)}
        self.episodes = []

        # Moving average windows
        self.short_window = 10
        self.long_window = 100

    def format_params(self, params):
        """Formats parameters into a readable string."""
        return '\n'.join([f"{key}: {value}" for key, value in params.items()])

    def calculate_moving_average(self, data, window_size):
        """Calculates the moving average of the rewards."""
        if len(data) < window_size:
            return np.array(data)
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')

    def plot_rewards(self, rewards_history):
        """Plots the reward progression and moving averages."""
        ax = self.axes[0, 0]  # Top-left
        ax.cla()

        ax.plot(rewards_history, label='Raw Reward', alpha=0.3, color='gray')

        if len(rewards_history) >= self.short_window:
            short_ma = self.calculate_moving_average(
                rewards_history, self.short_window)
            short_ma_episodes = np.arange(
                self.short_window-1, len(rewards_history))
            ax.plot(short_ma_episodes, short_ma,
                    label=f'{self.short_window}-Episode Moving Average',
                    color='blue', linewidth=2)

        if len(rewards_history) >= self.long_window:
            long_ma = self.calculate_moving_average(
                rewards_history, self.long_window)
            long_ma_episodes = np.arange(
                self.long_window-1, len(rewards_history))
            ax.plot(long_ma_episodes, long_ma,
                    label=f'{self.long_window}-Episode Moving Average',
                    color='red', linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True)

    def plot_sampling_distribution(self):
        """Plots the evolution of sampling distribution over episodes."""
        ax = self.axes[0, 1]  # Top-middle
        ax.cla()

        for env_id, samples in self.samples_history.items():
            if samples:
                total_samples = np.array(
                    [sum(s) for s in zip(*self.samples_history.values())])
                sample_percentages = np.array(samples) / total_samples * 100
                ax.plot(self.episodes, sample_percentages,
                        label=f'Env {env_id}', linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Samples (%)')
        ax.set_title('Sampling Distribution Over Time')
        ax.legend()
        ax.grid(True)
        ax.set_ylim([0, 100])

    def plot_done_ratio(self):
        """Plots the ratio of dones per episode per environment."""
        ax = self.axes[0, 2]  # Top-right
        ax.cla()

        for env_id, dones in self.done_history.items():
            if dones:
                cumulative_dones = np.cumsum(dones)
                episodes_array = np.array(self.episodes) + 1
                done_ratio = cumulative_dones / episodes_array
                ax.plot(self.episodes, done_ratio, label=f'Env {env_id}')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate (dones/episode)')
        ax.set_title('Environment Success Rate Over Time')
        ax.legend()
        ax.grid(True)
        ax.set_ylim([0, 1])

    def plot_episode_samples(self):
        """Plots the absolute number of samples per episode."""
        ax = self.axes[1, 0]  # Bottom-left
        ax.cla()

        for env_id, samples in self.samples_history.items():
            if samples:
                ax.plot(self.episodes, samples,
                        label=f'Env {env_id}', linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Samples per Episode')
        ax.legend()
        ax.grid(True)

    def plot_buffer_composition(self):
        """Plots the buffer composition over time."""
        ax = self.axes[1, 1]  # Bottom-middle
        ax.cla()

        for env_id, composition in self.buffer_composition_history.items():
            if composition:
                ax.plot(self.episodes, composition,
                        label=f'Env {env_id}', linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Buffer Composition (%)')
        ax.set_title('Replay Buffer Composition Over Time')
        ax.legend()
        ax.grid(True)
        ax.set_ylim([0, 100])

    def plot_buffer_absolute_counts(self):
        """Plots the absolute number of samples in buffer per environment."""
        ax = self.axes[1, 2]  # Bottom-right
        ax.cla()

        env_id_counts = {}
        for env_id, composition in self.buffer_composition_history.items():
            if composition:
                env_id_counts[env_id] = np.array(composition)

        for env_id, counts in env_id_counts.items():
            ax.plot(self.episodes, counts,
                    label=f'Env {env_id}', linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Buffer Sample Counts')
        ax.legend()
        ax.grid(True)

    def update(self, rewards_history, replay_buffer, episode, dones=None):
        """Updates all dashboard plots."""
        # Update done history
        if dones is not None:
            for env_id in self.done_history:
                self.done_history[env_id].append(
                    1 if dones.get(env_id, False) else 0)

        # Update samples history and buffer composition
        env_id_counts = replay_buffer.get_env_id_distribution()
        total_samples = sum(env_id_counts.values())

        for env_id in self.samples_history:
            # Update samples history
            count = env_id_counts.get(env_id, 0)
            if not self.samples_history[env_id]:
                self.samples_history[env_id].append(count)
            else:
                prev_count = self.samples_history[env_id][-1]
                new_samples = count - prev_count
                self.samples_history[env_id].append(new_samples)

            # Update buffer composition history
            composition_percentage = (
                count / total_samples * 100) if total_samples > 0 else 0
            self.buffer_composition_history[env_id].append(
                composition_percentage)

        self.episodes.append(episode)

        # Update all plots
        self.plot_rewards(rewards_history)
        self.plot_sampling_distribution()
        self.plot_done_ratio()
        self.plot_episode_samples()
        self.plot_buffer_composition()
        self.plot_buffer_absolute_counts()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.pause(0.01)

    def close(self):
        """Closes the dashboard."""
        plt.ioff()
        plt.show()
