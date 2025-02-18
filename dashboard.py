import numpy as np
import matplotlib.pyplot as plt


class Dashboard:
    def __init__(self, num_envs, params):
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        plt.ion()

        self.param_text = self.format_params_multicolumn(params, num_columns=3)
        self.fig.suptitle(self.param_text, fontsize=10)

        # Get engine info from parameters.
        self.engines_dict = params['Engines']

        self.env_keys = []
        global_index = 0
        for engine, count in self.engines_dict.items():
            for _ in range(count):
                self.env_keys.append(f"{engine}_{global_index}")
                global_index += 1

        # Initialize histories using the actual keys.
        self.done_history = {key: [] for key in self.env_keys}
        self.samples_history = {key: []
                                for key in self.env_keys}  # For sampling composition
        # For overall buffer composition
        self.buffer_composition_history = {key: [] for key in self.env_keys}

        self.buffer_fullness_history = []
        self.episodes = []
        self.short_window = 10
        self.long_window = 100

    def plot_sampling_composition(self):
        """
        Plots the sampling composition aggregated by engine type.
        (This uses data from self.samples_history, which is updated from the replay buffer's sampling distribution.)
        """
        ax = self.axes[0, 1]
        ax.cla()

        if any(self.samples_history.values()):
            # Group sampled counts by engine type
            engine_samples = {engine: []
                              for engine in self.engines_dict.keys()}
            for env_id, samples in self.samples_history.items():
                engine_type = env_id.split('_')[0]
                if not engine_samples[engine_type]:
                    engine_samples[engine_type] = samples
                else:
                    engine_samples[engine_type] = [sum(x) for x in zip(
                        engine_samples[engine_type], samples)]

            # Compute total sampled counts per episode
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

    def plot_done_ratio(self):
        ax = self.axes[0, 2]
        ax.cla()

        # Group success rates by engine type
        engine_dones = {engine: [] for engine in self.engines_dict.keys()}
        for env_id, dones in self.done_history.items():
            engine_type = env_id.split('_')[0]
            if dones:
                if not engine_dones[engine_type]:
                    engine_dones[engine_type] = dones
                else:
                    engine_dones[engine_type] = [
                        sum(x) / 2 for x in zip(engine_dones[engine_type], dones)]
        for engine_type, dones in engine_dones.items():
            if dones:
                cumulative_dones = np.cumsum(dones)
                episodes_array = np.array(self.episodes) + 1
                done_ratio = cumulative_dones / episodes_array
                ax.plot(self.episodes, done_ratio, label=f'{engine_type}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate by Engine Type')
        ax.legend()
        ax.grid(True)
        ax.set_ylim([0, 1])

    def update(self, rewards_history, replay_buffer, episode, episode_dones, stats):
        # Update done history per instance.
        if episode_dones is not None:
            for env_id in self.done_history:
                # Extract the global index from the identifier.
                env_num = int(env_id.split('_')[1])
                self.done_history[env_id].append(
                    1 if episode_dones.get(env_num, False) else 0)

        # Get overall buffer composition distribution from the replay buffer.
        env_id_counts = replay_buffer.get_env_id_distribution()
        total_buffer_samples = sum(env_id_counts.values())

        # Get sampling distribution (i.e. how many times experiences have been drawn) from the replay buffer.
        sampling_distribution = replay_buffer.get_sampling_distribution()
        total_sampled = sum(sampling_distribution.values())

        self.last_buffer_stats = replay_buffer.get_statistics()
        self.buffer_fullness_history.append(
            self.last_buffer_stats['fullness'] * 100)

        # Update histories for each instance.
        for env_id in self.samples_history:
            # Update sampling history from the new sampling distribution.
            count_sampled = sampling_distribution.get(env_id, 0)
            self.samples_history[env_id].append(count_sampled)
            # Update overall buffer composition history.
            count_buffer = env_id_counts.get(env_id, 0)
            composition_percentage = (
                count_buffer / total_buffer_samples * 100) if total_buffer_samples > 0 else 0
            self.buffer_composition_history[env_id].append(
                composition_percentage)

        self.episodes.append(episode)

        self.plot_rewards(rewards_history)
        self.plot_sampling_composition()
        self.plot_done_ratio()
        self.plot_episode_samples()
        self.plot_buffer_composition()
        # Plot for instance-level distribution.
        self.plot_buffer_instance_distribution()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.pause(0.01)

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
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True)

    def calculate_moving_average(self, data, window_size):
        if len(data) < window_size:
            return np.array(data)
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')

    def plot_episode_samples(self):
        ax = self.axes[1, 0]
        ax.cla()
        # Group samples by engine type (aggregated across instances)
        engine_samples = {engine: [] for engine in self.engines_dict.keys()}
        for env_id, samples in self.samples_history.items():
            engine_type = env_id.split('_')[0]
            if samples:
                if not engine_samples[engine_type]:
                    engine_samples[engine_type] = samples
                else:
                    engine_samples[engine_type] = [
                        a + b for a, b in zip(engine_samples[engine_type], samples)]
        for engine_type, samples in engine_samples.items():
            if samples:
                ax.plot(self.episodes, samples,
                        label=f'{engine_type}', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Samples per Episode by Engine Type')
        ax.legend()
        ax.grid(True)

    def plot_buffer_composition(self):
        ax = self.axes[1, 1]
        ax.cla()
        ax2 = ax.twinx()
        # Aggregate buffer composition by engine type.
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
        """
        This plot displays the distribution (as a percentage) for each individual environment instance.
        """
        ax = self.axes[1, 2]
        ax.cla()
        # For each instance, plot its composition history individually.
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
