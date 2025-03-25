from datetime import datetime
import numpy as np
import pandas as pd
import os
import csv
import time
import h5py
import h5py
import os
import numpy as np


class TrainingStats:
    def __init__(self, engines_dict):
        self.engines_dict = engines_dict
        self.unique_engines = list(engines_dict.keys())

        # Track rewards per engine type
        self.type_rewards = {engine: [] for engine in self.unique_engines}

        # Track timing per engine type
        self.start_times = {engine: None for engine in self.unique_engines}
        self.step_times = {engine: [] for engine in self.unique_engines}
        self.episode_durations = {engine: [] for engine in self.unique_engines}

        # Track steps per engine type
        self.episode_steps = {engine: [] for engine in self.unique_engines}

        # Create index mapping for dashboard compatibility
        self.engine_indices = {}
        self.agent_diagnostic = []

        env_id = 0
        for engine, count in engines_dict.items():
            self.engine_indices[engine] = []
            for _ in range(count):
                self.engine_indices[engine].append(env_id)
                env_id += 1

        # Track rewards per instance (for dashboard compatibility)
        self.instance_rewards = {i: []
                                 for i in range(sum(engines_dict.values()))}
        self.td_errors = []

    def start_instance_timing(self, engine):
        """Start timing for a specific engine"""
        self.start_times[engine] = time.time()

    def end_instance_timing(self, engine):
        """End timing for a specific engine and record duration"""
        if self.start_times[engine] is not None:
            duration = time.time() - self.start_times[engine]
            self.step_times[engine].append(duration)
            self.start_times[engine] = None

    def compute_episode_durations(self):
        """Compute average episode duration for each engine type"""
        for engine in self.unique_engines:
            if self.step_times[engine]:
                # Sum up all step times for this engine
                total_duration = sum(self.step_times[engine])
                self.episode_durations[engine].append(total_duration)
                # Clear the step times for next episode
                self.step_times[engine] = []
            else:
                # If no steps were recorded, use previous duration or 0
                last_duration = self.episode_durations[engine][-1] if self.episode_durations[engine] else 0
                self.episode_durations[engine].append(last_duration)

    def update_rewards(self, rewards_dict):
        """Update rewards for each engine type"""
        for engine, reward in rewards_dict.items():
            self.type_rewards[engine].append(reward)

            # Update instance rewards for dashboard compatibility
            for idx in self.engine_indices[engine]:
                self.instance_rewards[idx].append(reward)

    def update_steps(self, steps_dict):
        """Update steps for each engine type"""
        for engine, steps in steps_dict.items():
            self.episode_steps[engine].append(steps)

    def get_stats(self):
        """Get all statistics in a format compatible with the dashboard"""
        return {
            'instance': self.instance_rewards,
            'type': self.type_rewards,
            'episode_durations': self.episode_durations,
            'episode_steps': self.episode_steps,
            'td_errors': self.td_errors
        }

    def update_td_errors(self, td_errors):
        """Update TD errors for each training step"""
        self.td_errors.append(td_errors)

    def export_to_csv(self, output_dir='logs', filename=None):
        """
        Export ALL collected training statistics to CSV files with complete history.

        Args:
            output_dir (str): Directory to save CSV files
            filename (str, optional): Base filename. If None, a timestamp will be used
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamp for filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_stats_{timestamp}"

        # Export each type of data to its own CSV file

        # 1. Export rewards
        rewards_df = pd.DataFrame()
        for engine, rewards in self.type_rewards.items():
            rewards_df[f"{engine}_reward"] = pd.Series(rewards)
        rewards_df.to_csv(
            f"{output_dir}/{filename}_rewards.csv", index_label='episode')

        # 2. Export episode durations
        durations_df = pd.DataFrame()
        for engine, durations in self.episode_durations.items():
            durations_df[f"{engine}_duration"] = pd.Series(durations)
        durations_df.to_csv(
            f"{output_dir}/{filename}_durations.csv", index_label='episode')

        # 3. Export step times (raw step timing data)
        step_times_df = pd.DataFrame()
        for engine, times in self.step_times.items():
            step_times_df[f"{engine}_step_times"] = pd.Series(times)
        step_times_df.to_csv(
            f"{output_dir}/{filename}_step_times.csv", index_label='step')

        # 4. Export episode steps
        steps_df = pd.DataFrame()
        for engine, steps in self.episode_steps.items():
            steps_df[f"{engine}_steps"] = pd.Series(steps)
        steps_df.to_csv(f"{output_dir}/{filename}_steps.csv",
                        index_label='episode')

        # 5. Export instance rewards
        instance_df = pd.DataFrame()
        for instance_id, rewards in self.instance_rewards.items():
            instance_df[f"instance_{instance_id}_reward"] = pd.Series(rewards)
        instance_df.to_csv(
            f"{output_dir}/{filename}_instance_rewards.csv", index_label='episode')

        # 6. Export TD errors - handling the nested list structure
        if self.td_errors:
            # Flatten the nested list structure
            episode_indices = []
            flattened_td_errors = []

            for episode_idx, errors_list in enumerate(self.td_errors):
                for error in errors_list:
                    episode_indices.append(episode_idx)
                    flattened_td_errors.append(error)

            td_df = pd.DataFrame({
                'episode': episode_indices,
                'td_error': flattened_td_errors
            })
            td_df.to_csv(f"{output_dir}/{filename}_td_errors.csv", index=False)

        print(
            f"✅ All training statistics exported to {output_dir}/{filename}_*.csv")

    def export_to_hdf5(self, output_dir='./logs', download_local=True, save_to_drive=False):
        """
        Export training statistics to HDF5 with options for Colab workflow

        Args:
            output_dir: Directory to save in Colab VM
            download_local: Whether to download the file locally after saving
            save_to_drive: Whether to also save to Google Drive

        Returns:
            Path to the saved file
        """

        # Check if running in Colab
        in_colab = 'google.colab' in str(get_ipython())

        # Create local directory in Colab VM
        os.makedirs(output_dir, exist_ok=True)

        # Extract environment name
        env_name = self.env_name if hasattr(
            self, 'env_name') else "unknown_env"

        # Extract sampling composition
        sampling_info = ""
        if hasattr(self, 'agent') and hasattr(self.agent, 'replay_buffer') and hasattr(self.agent.replay_buffer, 'sampling_composition'):
            composition = self.agent.replay_buffer.sampling_composition
            for engine, percentage in composition.items():
                sampling_info += f"{engine}{int(percentage*100)}_"
            sampling_info = sampling_info.rstrip('_')
        elif hasattr(self, 'replay_buffer') and hasattr(self.replay_buffer, 'sampling_composition'):
            composition = self.replay_buffer.sampling_composition
            for engine, percentage in composition.items():
                sampling_info += f"{engine}{int(percentage*100)}_"
            sampling_info = sampling_info.rstrip('_')

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create descriptive filename
        filename = f"{env_name}_{sampling_info}_{timestamp}.h5"
        filepath = f"{output_dir}/{filename}"

        # Create the HDF5 file
        with h5py.File(filepath, 'w') as f:
            # Create main groups
            rewards_group = f.create_group('rewards')
            durations_group = f.create_group('durations')
            steps_group = f.create_group('steps')
            instance_rewards_group = f.create_group('instance_rewards')
            td_errors_group = f.create_group('td_errors')
            metadata_group = f.create_group('metadata')

            # Add compression options for numerical data
            compression_opts = {
                'compression': 'gzip',
                'compression_opts': 4  # Compression level
            }

            # Store rewards data with attributes
            for engine, rewards in self.type_rewards.items():
                ds = rewards_group.create_dataset(engine, data=np.array(
                    rewards, dtype=np.float32), **compression_opts)
                # Add useful attributes
                if len(rewards) > 0:
                    ds.attrs['mean'] = np.mean(rewards)
                    ds.attrs['max'] = np.max(rewards)
                    ds.attrs['min'] = np.min(rewards)
                    ds.attrs['std'] = np.std(rewards)

            # Store durations data
            for engine, durations in self.episode_durations.items():
                ds = durations_group.create_dataset(engine, data=np.array(
                    durations, dtype=np.float32), **compression_opts)
                if len(durations) > 0:
                    ds.attrs['mean'] = np.mean(durations)
                    ds.attrs['max'] = np.max(durations)

            # Store steps data
            for engine, steps in self.episode_steps.items():
                ds = steps_group.create_dataset(engine, data=np.array(
                    steps, dtype=np.int32), **compression_opts)
                if len(steps) > 0:
                    ds.attrs['mean'] = np.mean(steps)
                    ds.attrs['max'] = np.max(steps)

            # Store instance rewards data
            for instance_id, rewards in self.instance_rewards.items():
                ds = instance_rewards_group.create_dataset(
                    f"instance_{instance_id}",
                    data=np.array(rewards, dtype=np.float32),
                    **compression_opts
                )

            # Store TD errors (special handling for nested structure)
            for episode_idx, errors in enumerate(self.td_errors):
                if errors:  # Only create dataset if there are errors
                    td_errors_group.create_dataset(
                        f"episode_{episode_idx}",
                        data=np.array(errors, dtype=np.float32),
                        **compression_opts
                    )

            # Add metadata
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata_group.create_dataset(
                'timestamp', data=np.string_(timestamp_str))

            # Store engines configuration
            engines_str = str(self.engines_dict)
            metadata_group.create_dataset(
                'engines', data=np.string_(engines_str))

            # Store sampling composition in metadata if available
            if sampling_info:
                metadata_group.create_dataset(
                    'sampling_composition', data=np.string_(sampling_info))

            # Store total episodes
            num_episodes = len(next(iter(self.type_rewards.values())))
            metadata_group.create_dataset('num_episodes', data=num_episodes)

        print(f"✅ Statistics saved to {filepath}")

        # If in Colab and download requested, download to local machine
        if in_colab and download_local:
            try:
                files.download(filepath)
                print(f"✅ File downloaded to your local machine: {filename}")
            except Exception as e:
                print(f"⚠️ Error downloading file: {e}")

        # If in Colab and Google Drive backup requested
        if in_colab and save_to_drive:
            try:
                from google.colab import drive

                # Check if drive is mounted, if not, mount it
                mounted = os.path.exists('/content/drive')
                if not mounted:
                    drive.mount('/content/drive')

                # Create directory in Drive
                drive_dir = f"/content/drive/MyDrive/RL_Experiments/{env_name}/{sampling_info}"
                os.makedirs(drive_dir, exist_ok=True)

                # Copy file to Drive
                drive_path = f"{drive_dir}/{filename}"
                import shutil
                shutil.copyfile(filepath, drive_path)

                print(f"✅ Backup saved to Google Drive: {drive_path}")
            except Exception as e:
                print(f"⚠️ Error saving to Google Drive: {e}")

        return filepath
