import numpy as np
import matplotlib.pyplot as plt


class Dashboard:
    def __init__(self, num_envs, params):
        self.fig, self.axes = plt.subplots(2, 4, figsize=(18, 10))
        plt.ion()

        self.param_text = self.format_params_multicolumn(params, num_columns=3)
        self.fig.suptitle(self.param_text, fontsize=10)

        self.engines_dict = params['Engines']

        self.env_keys = []
        global_index = 0
        for engine, count in self.engines_dict.items():
            for _ in range(count):
                self.env_keys.append(f"{engine}_{global_index}")
                global_index += 1

        # Initialize histories
        self.samples_history = {key: [] for key in self.env_keys}
        self.buffer_composition_history = {key: [] for key in self.env_keys}

        # New: Track episode timing history and sample ages
        self.episode_times_history = {engine: []
                                      for engine in self.engines_dict.keys()}
        self.sample_ages_history = {engine: []
                                    for engine in self.engines_dict.keys()}

        self.buffer_fullness_history = []
        self.episodes = []
        self.short_window = 10
        self.long_window = 100
        self.episode_steps_history = {engine: []
                                      for engine in self.engines_dict.keys()}
        self.type_rewards = {engine: [] for engine in self.engines_dict.keys()}

    def plot_rewards(self, rewards_history):
        ax = self.axes[0, 0]
        ax.cla()
        ax.plot(rewards_history, label='Raw Reward', alpha=0.3, color='gray')
        if len(rewards_history) >= self.short_window:
            short_ma = self.calculate_moving_average(
                rewards_history, self.short_window)
            short_ma_episodes = np.arange(
                self.short_window - 1, len(rewards_history))
            ax.plot(short_ma_episodes, short_ma,
                    label=f'{self.short_window}-Episode Moving Average',
                    color='blue', linewidth=2)
        if len(rewards_history) >= self.long_window:
            long_ma = self.calculate_moving_average(
                rewards_history, self.long_window)
            long_ma_episodes = np.arange(
                self.long_window - 1, len(rewards_history))
            ax.plot(long_ma_episodes, long_ma,
                    label=f'{self.long_window}-Episode Moving Average',
                    color='red', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Weighted Average Reward')
        ax.legend()
        ax.grid(True)

    def plot_sampling_composition(self):
        ax = self.axes[1, 3]
        ax.cla()

        if any(self.samples_history.values()):
            engine_samples = {engine: []
                              for engine in self.engines_dict.keys()}
            for env_id, samples in self.samples_history.items():
                engine_type = env_id.split('_')[0]
                if not engine_samples[engine_type]:
                    engine_samples[engine_type] = samples
                else:
                    engine_samples[engine_type] = [sum(x) for x in zip(
                        engine_samples[engine_type], samples)]

            total_samples = np.array([sum(s)
                                     for s in zip(*engine_samples.values())])
            for engine_type, samples in engine_samples.items():
                if samples:
                    sample_percentages = np.divide(samples, total_samples,
                                                   out=np.zeros_like(
                                                       samples, dtype=float),
                                                   where=total_samples != 0) * 100
                    current_percentage = sample_percentages[-1] if len(
                        sample_percentages) > 0 else 0
                    ax.plot(self.episodes, sample_percentages,
                            label=f'{engine_type} ({current_percentage:.1f}%)',
                            linewidth=3)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Samples (%)')
        ax.set_title('Sampling Composition by Engine Type')
        ax.legend()
        ax.grid(True)
        ax.set_ylim([0, 100])

    def plot_average_sample_age(self):
        ax = self.axes[1, 2]
        ax.cla()

        for engine_type, ages in self.sample_ages_history.items():
            if ages:
                current_age = ages[-1] if ages else 0
                ax.plot(self.episodes, ages,
                        label=f'{engine_type} ({current_age:.1f} episodes)',
                        linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Experience Age (episodes)')
        ax.set_title('Average Experience Age by Engine Type')
        ax.legend()
        ax.grid(True)

    def plot_episode_times(self):
        """
        Plots the moving average (10 episodes) of time taken per episode for each engine type.
        """
        ax = self.axes[0, 3]
        ax.cla()

        window_size = 10  # Moving average window

        # Get a color iterator from matplotlib's default color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_idx = 0

        for engine_type, times in self.episode_times_history.items():
            if times:
                current_color = colors[color_idx %
                                       len(colors)]  # Cycle through colors

                # Calculate moving average if we have enough data
                if len(times) >= window_size:
                    moving_avg = self.calculate_moving_average(
                        times, window_size)
                    ma_episodes = np.arange(window_size - 1, len(times))
                    current_avg = moving_avg[-1] if len(moving_avg) > 0 else 0

                    # Plot both raw data and moving average with the same color
                    ax.plot(self.episodes, times, alpha=0.3,
                            label=f'{engine_type} (raw)', linewidth=1,
                            color=current_color)
                    ax.plot(ma_episodes, moving_avg,
                            label=f'{engine_type} ({current_avg:.2f}s avg)',
                            linewidth=2, color=current_color)
                else:
                    # If we don't have enough data for moving average, just plot raw data
                    current_time = times[-1] if times else 0
                    ax.plot(self.episodes, times,
                            label=f'{engine_type} ({current_time:.2f}s)',
                            linewidth=2, color=current_color)

                color_idx += 1
        ax.set_xlabel('Episode')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(
            f'Episode Time by Engine Type ({window_size}-Episode Moving Average)')
        ax.legend()
        ax.grid(True)

    def plot_buffer_composition(self):
        ax = self.axes[1, 0]
        ax.cla()
        ax2 = ax.twinx()

        engine_composition = {engine: []
                              for engine in self.engines_dict.keys()}
        for env_id, composition in self.buffer_composition_history.items():
            engine_type = env_id.split('_')[0]
            if composition:
                if not engine_composition[engine_type]:
                    engine_composition[engine_type] = composition
                else:
                    engine_composition[engine_type] = [
                        a + b for a, b in zip(engine_composition[engine_type], composition)]

        for engine_type, composition in engine_composition.items():
            if composition:
                current_percentage = composition[-1] if composition else 0
                ax.plot(self.episodes, composition,
                        label=f'{engine_type} ({current_percentage:.1f}%)', linewidth=3)

        if self.buffer_fullness_history:
            current_fullness = self.buffer_fullness_history[-1]
            stats = self.last_buffer_stats
            current_size = stats['size']
            total_capacity = stats['capacity']
            fullness_label = f'Buffer Fullness: {current_fullness:.1f}% ({current_size:,}/{total_capacity:,})'
            ax2.plot(self.episodes, self.buffer_fullness_history,
                     label=fullness_label, color='black', linestyle='--', linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Buffer Composition (%)')
        ax2.set_ylabel('Buffer Fullness (%)')
        ax.set_title('Replay Buffer Composition by Engine Type')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax.grid(True)
        ax.set_ylim([0, 100])
        ax2.set_ylim([0, 100])

    def plot_buffer_instance_distribution(self):
        ax = self.axes[1, 1]
        ax.cla()
        for env_id, composition in self.buffer_composition_history.items():
            if composition:
                ax.plot(self.episodes, composition,
                        label=f'{env_id}', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Distribution (%)')
        ax.set_title('Buffer Composition by Instance')
        ax.legend()
        ax.grid(True)
        ax.set_ylim([0, 100])

    def update(self, rewards_history, replay_buffer, episode, episode_dones, stats):
        # Get replay buffer statistics
        buffer_stats = replay_buffer.get_statistics()

        self.type_rewards = stats.get_stats()['type']

        # Update buffer fullness
        self.buffer_fullness_history.append(buffer_stats['fullness'] * 100)
        self.last_buffer_stats = buffer_stats

        # Update histories for each instance
        env_id_counts = replay_buffer.get_env_id_distribution()
        total_buffer_samples = sum(env_id_counts.values())
        sampling_distribution = replay_buffer.get_sampling_distribution()

        for env_id in self.samples_history:
            # Update sampling history
            count_sampled = sampling_distribution.get(env_id, 0)
            self.samples_history[env_id].append(count_sampled)

            # Update buffer composition history
            count_buffer = env_id_counts.get(env_id, 0)
            composition_percentage = (
                count_buffer / total_buffer_samples * 100) if total_buffer_samples > 0 else 0
            self.buffer_composition_history[env_id].append(
                composition_percentage)

        # Update episode timing history
        if hasattr(stats, 'episode_durations'):
            for engine_type, durations in stats.episode_durations.items():
                if durations:
                    self.episode_times_history[engine_type].append(
                        durations[-1])

        # Update sample ages from buffer statistics
        if 'sample_ages' in buffer_stats:
            for engine_type, age in buffer_stats['sample_ages'].items():
                self.sample_ages_history[engine_type].append(age)

        if hasattr(stats, 'episode_steps'):
            for engine_type, steps in stats.episode_steps.items():
                if steps:
                    self.episode_steps_history[engine_type].append(steps[-1])
        self.episodes.append(episode)

        # Update all plots
        self.plot_rewards(rewards_history)
        self.plot_sampling_composition()
        self.plot_average_sample_age()
        self.plot_episode_times()
        self.plot_buffer_composition()
        self.plot_buffer_instance_distribution()
        self.plot_episode_steps()
        self.plot_rewards_by_engine()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.pause(0.01)

    def calculate_moving_average(self, data, window_size):
        if len(data) < window_size:
            return np.array(data)
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')

    def format_params_multicolumn(self, params, num_columns=3):
        param_list = [f"{key} {value}" for key, value in params.items()]
        num_rows = (len(param_list) + num_columns - 1) // num_columns
        while len(param_list) < num_rows * num_columns:
            param_list.append('')
        columns = []
        for col in range(num_columns):
            column = param_list[col::num_columns]
            columns.append('\n'.join(column))
        return '     |     '.join(columns)

    def close(self):
        plt.ioff()
        plt.show()

    def plot_episode_steps(self):
        """
        Plots the steps taken per episode for each engine type, with individual points and
        a 10-episode moving average.
        """
        ax = self.axes[0, 2]  # Using the last unused subplot
        ax.cla()

        window_size = 10  # Moving average window
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_idx = 0

        for engine_type, steps in self.episode_steps_history.items():
            if steps:
                current_color = colors[color_idx % len(colors)]

                if len(steps) >= window_size:
                    moving_avg = self.calculate_moving_average(
                        steps, window_size)
                    ma_episodes = np.arange(window_size - 1, len(steps))
                    current_avg = moving_avg[-1] if len(moving_avg) > 0 else 0

                    # Plot individual steps as scatter points
                    ax.scatter(self.episodes, steps, alpha=0.3,
                               label=f'{engine_type} (steps)', color=current_color, s=20)
                    # Plot moving average as solid line
                    ax.plot(ma_episodes, moving_avg,
                            label=f'{engine_type} ({current_avg:.0f} avg)',
                            linewidth=2, color=current_color)
                else:
                    current_steps = steps[-1] if steps else 0
                    ax.scatter(self.episodes, steps,
                               label=f'{engine_type} ({current_steps:.0f})',
                               color=current_color, s=20)

                color_idx += 1

        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title(
            f'Steps per Episode ({window_size}-Episode Moving Average)')
        ax.legend()
        ax.grid(True)

    def plot_rewards_by_engine(self):
        """
        Plots the rewards grouped by engine type, showing raw data as dots and moving averages as lines.
        """
        ax = self.axes[0, 1]  # Using the remaining subplot in the top row
        ax.cla()

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_idx = 0

        for engine_type, rewards in self.type_rewards.items():
            if rewards:
                current_color = colors[color_idx % len(colors)]

                # Plot raw rewards as scatter points with low alpha
                ax.scatter(self.episodes, rewards, alpha=0.3,
                           # s=20 sets the dot size
                           label=f'{engine_type} (raw)', s=20,
                           color=current_color)

                # Calculate and plot moving averages if enough data
                if len(rewards) >= self.short_window:
                    short_ma = self.calculate_moving_average(
                        rewards, self.short_window)
                    short_ma_episodes = np.arange(
                        self.short_window - 1, len(rewards))
                    ax.plot(short_ma_episodes, short_ma,
                            label=f'{engine_type} ({self.short_window}-ep avg)',
                            color=current_color, linewidth=2)

                color_idx += 1

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Average Reward by Engine Type')
        ax.legend()
        ax.grid(True)
