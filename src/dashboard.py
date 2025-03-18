import numpy as np
import matplotlib.pyplot as plt


class Dashboard:
    def __init__(self, num_envs, params):
        self.fig, self.axes = plt.subplots(2, 4, figsize=(18, 10))
        plt.ion()

        self.param_text = self.format_params_multicolumn(params, num_columns=3)
        self.fig.suptitle(self.param_text, fontsize=10)

        self.engines_dict = params['Engines']

        # Simplified tracking for sequential training
        self.samples_history = {engine: []
                                for engine in self.engines_dict.keys()}
        self.buffer_composition_history = {engine: []
                                           for engine in self.engines_dict.keys()}

        # Existing tracking for engine-level metrics
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
        self.td_error_history = []

    def update(self, rewards_history, replay_buffer, episode, episode_dones, stats):
        # Get sampling distribution
        sampling_distribution = replay_buffer.get_sampling_distribution()

        # Update samples history by engine type
        for engine_type in self.samples_history:
            # Sum samples for this engine type
            engine_samples = sum(
                count for env_id, count in sampling_distribution.items()
                if env_id.startswith(engine_type)
            )

            # Append total samples for this engine type
            self.samples_history[engine_type].append(engine_samples)

        # Rest of the existing update method remains the same
        buffer_stats = replay_buffer.get_statistics()

        self.type_rewards = stats.get_stats()['type']

        # Update buffer fullness
        self.buffer_fullness_history.append(buffer_stats['fullness'] * 100)
        self.last_buffer_stats = buffer_stats

        # Update histories for each instance
        env_id_counts = replay_buffer.get_env_id_distribution()
        total_buffer_samples = sum(env_id_counts.values())
        sampling_distribution = replay_buffer.get_sampling_distribution()

        for engine_type in self.buffer_composition_history:
            # Find experiences for this engine type
            engine_experiences = sum(
                count for env_id, count in env_id_counts.items()
                if env_id.startswith(engine_type)
            )

            # Calculate composition percentage
            composition_percentage = (
                engine_experiences / total_buffer_samples * 100
            ) if total_buffer_samples > 0 else 0

            # Update buffer composition history
            self.buffer_composition_history[engine_type].append(
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

        self.td_error_history = stats.get_stats()['td_errors']

        self.episodes.append(episode)

        # Update all plots
        self.plot_rewards(rewards_history)
        self.plot_sampling_composition()
        self.plot_average_sample_age()
        self.plot_episode_times()
        self.plot_buffer_composition()
        self.plot_episode_steps()
        self.plot_rewards_by_engine()
        self.plot_td_error()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.pause(0.01)

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
        """
        Plot sampling composition with comprehensive debugging
        """
        ax = self.axes[1, 3]
        ax.cla()

        # Verify sampling history exists
        if not any(self.samples_history.values()):
            ax.set_title('Sampling Composition (No Data)')
            return

        # Aggregate samples by engine type
        engine_samples = {engine: [] for engine in self.engines_dict.keys()}

        # Collect samples for each engine type
        for env_id, samples in self.samples_history.items():
            engine_type = env_id.split('_')[0]

            if engine_type not in engine_samples:
                print(f"⚠️ Unexpected engine type: {engine_type}")
                continue

            # If samples list exists, use it
            if samples:
                if not engine_samples[engine_type]:
                    engine_samples[engine_type] = samples
                else:
                    # Sum samples across same engine type
                    engine_samples[engine_type] = [
                        sum(x) for x in zip(engine_samples[engine_type], samples)
                    ]

        # Compute total samples
        total_samples = np.array([sum(s)
                                 for s in zip(*engine_samples.values())])

        # Plot sampling composition
        for engine_type, samples in engine_samples.items():
            if samples:
                # Calculate percentages
                sample_percentages = np.divide(
                    samples,
                    total_samples,
                    out=np.zeros_like(samples, dtype=float),
                    where=total_samples != 0
                ) * 100

                # Get current percentage for legend
                current_percentage = sample_percentages[-1] if len(
                    sample_percentages) > 0 else 0

                # Plot the sampling composition
                ax.plot(
                    self.episodes[:len(sample_percentages)],
                    sample_percentages,
                    label=f'{engine_type} ({current_percentage:.1f}%)',
                    linewidth=3
                )

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
                current_color = colors[color_idx % len(colors)]

                # Calculate moving average if we have enough data
                if len(times) >= window_size:
                    moving_avg = self.calculate_moving_average(
                        times, window_size)
                    ma_episodes = np.arange(window_size - 1, len(times))
                    current_avg = moving_avg[-1] if len(moving_avg) > 0 else 0

                    # Plot raw data as scatter points
                    ax.scatter(self.episodes, times, alpha=0.3,
                               label=f'{engine_type} (raw)',
                               color=current_color, s=20)

                    # Plot moving average as solid line
                    ax.plot(ma_episodes, moving_avg,
                            label=f'{engine_type} ({current_avg:.2f}s avg)',
                            linewidth=2, color=current_color)
                else:
                    # If we don't have enough data for moving average, just plot raw data
                    current_time = times[-1] if times else 0
                    ax.scatter(self.episodes, times,
                               label=f'{engine_type} ({current_time:.2f}s)',
                               color=current_color, s=20)

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
        plt.show(block=True)

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

    # In dashboard.py, add this method:

    def plot_td_error(self):
        """Plot TD error over episodes with individual points and trend line"""
        ax = self.axes[1, 1]
        ax.cla()

        # Check if td_error_history exists and is not empty
        if not self.td_error_history or len(self.td_error_history) == 0:
            ax.set_title('TD Error over Episodes (No Data)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('TD Error')
            ax.grid(True)
            return

        # Simple case: one average TD error value per episode
        if isinstance(self.td_error_history[0], (int, float, np.number)):
            # Make sure we have the same number of episodes and TD errors
            min_len = min(len(self.episodes), len(self.td_error_history))

            ax.plot(self.episodes[:min_len], self.td_error_history[:min_len],
                    label='Mean TD Error', color='purple', linewidth=2)

            # Add moving average if we have enough data
            if min_len >= self.short_window:
                short_ma = self.calculate_moving_average(
                    self.td_error_history[:min_len], self.short_window)
                short_ma_episodes = np.arange(
                    self.short_window - 1, min_len)
                ax.plot(short_ma_episodes, short_ma,
                        label=f'{self.short_window}-Episode MA',
                        color='magenta', linewidth=2)

        # If you've been appending arrays or lists, we need to handle differently
        elif isinstance(self.td_error_history[0], (list, np.ndarray)):
            # Calculate episode-wise means for the line plot
            episode_means = []
            for errors in self.td_error_history:
                if len(errors) > 0:
                    episode_means.append(float(np.mean(errors)))
                else:
                    episode_means.append(0.0)  # Handle empty arrays

            # Make sure we have the same number of episodes and means
            min_len = min(len(self.episodes), len(episode_means))

            # Plot the line of means
            ax.plot(self.episodes[:min_len], episode_means[:min_len],
                    label='Mean TD Error', color='purple', linewidth=2)

            # For scatter plot, we need to flatten everything
            all_points_x = []
            all_points_y = []

            for episode_idx, errors in enumerate(self.td_error_history[:min_len]):
                if len(errors) == 0:
                    continue  # Skip empty arrays

                # Convert episode index to float for jitter operation
                episode_x = np.full(len(errors), float(
                    self.episodes[episode_idx]))
                episode_x = episode_x + \
                    np.random.normal(0, 0.1, len(errors))  # Small jitter

                all_points_x.extend(episode_x)
                all_points_y.extend(errors)

            # Only plot scatter if we have points to plot
            if len(all_points_x) > 0:
                # Plot individual points with low alpha for density visualization
                ax.scatter(all_points_x, all_points_y, alpha=0.1, color='blue',
                           s=5, label='Individual Errors')

            # Add moving average if we have enough data
            if min_len >= self.short_window:
                short_ma = self.calculate_moving_average(
                    episode_means[:min_len], self.short_window)
                short_ma_episodes = np.arange(
                    self.short_window - 1, min_len)
                ax.plot(short_ma_episodes, short_ma,
                        label=f'{self.short_window}-Episode MA',
                        color='magenta', linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('TD Error')
        ax.set_title('TD Error over Episodes')
        ax.legend()
        ax.grid(True)
