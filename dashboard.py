import numpy as np
import matplotlib.pyplot as plt


class Dashboard:
    def __init__(self, num_envs, params):
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        plt.ion()

        # Format parameters in multiple columns and display at the top
        param_text = self.format_params_multicolumn(params, num_columns=3)
        self.fig.suptitle(param_text, fontsize=10)

        self.done_history = {env_id: [] for env_id in range(num_envs)}
        self.samples_history = {env_id: [] for env_id in range(num_envs)}
        self.buffer_composition_history = {
            env_id: [] for env_id in range(num_envs)}
        self.buffer_fullness_history = []  # Track buffer fullness
        self.episodes = []

        self.short_window = 10
        self.long_window = 100

    def format_params_multicolumn(self, params, num_columns=3):
        """Formats parameters into multiple columns."""
        # Convert params into list of key-value pairs
        param_list = [f"{key} {value}" for key, value in params.items()]

        # Calculate rows needed
        num_params = len(param_list)
        num_rows = (num_params + num_columns - 1) // num_columns

        # Pad the list to fill all columns
        while len(param_list) < num_rows * num_columns:
            param_list.append('')

        # Create columns
        columns = []
        for col in range(num_columns):
            column = param_list[col::num_columns]
            columns.append('\n'.join(column))

        # Join columns with spacing
        return '     |     '.join(columns)

    def calculate_moving_average(self, data, window_size):
        if len(data) < window_size:
            return np.array(data)
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')

    def plot_rewards(self, rewards_history):
        ax = self.axes[0, 0]
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

    def plot_sampling_composition(self):
        """Plots the evolution of sampling composition over episodes."""
        ax = self.axes[0, 1]
        ax.cla()

        # Plot sampling composition
        for env_id, samples in self.samples_history.items():
            if samples:
                total_samples = np.array(
                    [sum(s) for s in zip(*self.samples_history.values())])
                sample_percentages = np.array(samples) / total_samples * 100
                ax.plot(self.episodes, sample_percentages,
                        label=f'Env {env_id}', linewidth=3)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Samples (%)')
        ax.set_title('Sampling Composition Over Time')
        ax.legend()
        ax.grid(True)
        ax.set_ylim([0, 100])

    def plot_done_ratio(self):
        ax = self.axes[0, 2]
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
        ax = self.axes[1, 0]
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
        """Plots the buffer composition over time with detailed buffer fullness."""
        ax = self.axes[1, 1]
        ax.cla()

        # Create secondary y-axis for buffer fullness
        ax2 = ax.twinx()

        # Plot buffer composition on primary y-axis
        for env_id, composition in self.buffer_composition_history.items():
            if composition:
                ax.plot(self.episodes, composition,
                        label=f'Env {env_id}', linewidth=3)

        # Plot buffer fullness on secondary y-axis
        if self.buffer_fullness_history:
            current_fullness = self.buffer_fullness_history[-1]
            # Get the absolute number from the replay buffer statistics
            stats = self.last_buffer_stats  # Stored during update
            current_size = stats['size']
            total_capacity = stats['capacity']

            fullness_label = f'Buffer Fullness: {current_fullness:.1f}% ({current_size:,}/{total_capacity:,})'

            ax2.plot(self.episodes, self.buffer_fullness_history,
                     label=fullness_label,
                     color='black', linestyle='--', linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Buffer Composition (%)')
        ax2.set_ylabel('Buffer Fullness (%)')
        ax.set_title('Replay Buffer Composition Over Time')

        # Adjust legends - combine both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        ax.grid(True)
        ax.set_ylim([0, 100])
        ax2.set_ylim([0, 100])

    def plot_buffer_absolute_counts(self):
        ax = self.axes[1, 2]
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
        if dones is not None:
            for env_id in self.done_history:
                self.done_history[env_id].append(
                    1 if dones.get(env_id, False) else 0)

        env_id_counts = replay_buffer.get_env_id_distribution()
        total_samples = sum(env_id_counts.values())

        # Store buffer statistics for use in plotting
        self.last_buffer_stats = replay_buffer.get_statistics()
        self.buffer_fullness_history.append(
            self.last_buffer_stats['fullness'] * 100)  # Convert to percentage

        for env_id in self.samples_history:
            count = env_id_counts.get(env_id, 0)
            if not self.samples_history[env_id]:
                self.samples_history[env_id].append(count)
            else:
                prev_count = self.samples_history[env_id][-1]
                new_samples = count - prev_count
                self.samples_history[env_id].append(new_samples)

            composition_percentage = (
                count / total_samples * 100) if total_samples > 0 else 0
            self.buffer_composition_history[env_id].append(
                composition_percentage)

        self.episodes.append(episode)

        # Update all plots
        self.plot_rewards(rewards_history)
        self.plot_sampling_composition()
        self.plot_done_ratio()
        self.plot_episode_samples()
        self.plot_buffer_composition()
        self.plot_buffer_absolute_counts()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.pause(0.01)

    def close(self):
        plt.ioff()
        plt.show()
