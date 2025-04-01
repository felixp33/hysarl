import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob


def sliding_window_avg(data, window=11):
    """Apply a sliding window average to smooth the data"""
    return np.convolve(data, np.ones(window) / window, mode='valid')


class MetricsVisualizer:
    def __init__(self, window_size=11):
        self.window_size = window_size
        self.colors = {
            "mujoco": "blue",
            "brax": "orange",
        }

        # Initialize data containers
        self.composition = {}
        self.data = {}
        self.timesteps = None
        self.experiment = None
        self.loaded_files = []

    def get(self, composition, experiment=None, file_path=None, data_dir=None):
        """Load metrics data based on composition dictionary and experiment name

        Args:
            composition: Dict mapping algorithm names to weights (e.g., {'mujoco': 0.8, 'brax': 0.2})
            experiment: Experiment name (e.g., 'HalfCheetah', 'Walker2d')
            file_path: Direct path to a CSV file (optional)
            data_dir: Optional directory to search for CSV files

        Returns:
            self, for method chaining
        """
        self.composition = composition
        if experiment:
            self.experiment = experiment.lower()
        self.data = {}

        # If direct file path is provided, use it
        if file_path and os.path.exists(file_path):
            self.loaded_files.append(file_path)
            self._load_file(file_path)
            return self

        # Otherwise, search for matching files based on experiment and composition
        if not experiment:
            raise ValueError("Either file_path or experiment must be provided")

        # Extract algorithm weights
        mujoco_weight = int(composition.get('mujoco', 0) * 100)
        brax_weight = int(composition.get('brax', 0) * 100)

        # Create file pattern
        file_pattern = f"{self.experiment}_m{mujoco_weight}_b{brax_weight}_*.csv"

        # Add data directory if provided
        search_path = os.path.join(
            data_dir, file_pattern) if data_dir else file_pattern

        # Find matching files
        matching_files = glob.glob(search_path)

        if matching_files:
            # Use the first matching file
            file_path = matching_files[0]
            self.loaded_files.append(file_path)
            self._load_file(file_path)
        else:
            raise FileNotFoundError(
                f"No CSV files found matching pattern '{search_path}'")

        return self

    def _load_file(self, file_path):
        """Load and process a CSV file with expected column names:
           episode, mujoco_reward, brax_reward
        """
        try:
            # Load CSV data
            df = pd.read_csv(file_path)

            # Check for required columns
            required_columns = ['episode', 'mujoco_reward', 'brax_reward']
            missing_columns = [
                col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(
                    f"CSV file is missing required columns: {missing_columns}")

            # Extract data from columns
            episodes = df['episode'].values

            # Process data for each algorithm in the composition
            if 'mujoco' in self.composition:
                mujoco_rewards = df['mujoco_reward'].values
                self._process_algorithm_data(
                    'mujoco', episodes, mujoco_rewards)

            if 'brax' in self.composition:
                brax_rewards = df['brax_reward'].values
                self._process_algorithm_data('brax', episodes, brax_rewards)

        except Exception as e:
            raise Exception(f"Error loading {file_path}: {str(e)}")

    def _process_algorithm_data(self, algo_name, episodes, rewards):
        """Process data for a single algorithm"""
        # Apply smoothing
        if len(rewards) >= self.window_size:
            smoothed_rewards = sliding_window_avg(rewards, self.window_size)

            # Calculate rolling std for the shaded area
            rolling_std = []
            for i in range(len(rewards) - self.window_size + 1):
                window = rewards[i:i+self.window_size]
                rolling_std.append(np.std(window))
            rolling_std = np.array(rolling_std)

            # Adjusted timesteps to match smoothed array length
            offset = (self.window_size - 1) // 2
            episodes_smooth = episodes[offset:-(offset)]
        else:
            # Not enough data points for smoothing, use raw data
            smoothed_rewards = rewards
            rolling_std = np.zeros_like(rewards)
            episodes_smooth = episodes

        # Store the processed data
        self.data[algo_name] = {
            'episodes': episodes,
            'rewards': rewards,
            'smoothed_rewards': smoothed_rewards,
            'std': rolling_std,
            'episodes_smooth': episodes_smooth
        }

        # Set the episodes reference if not already set
        if self.timesteps is None and len(episodes_smooth) > 0:
            self.timesteps = episodes_smooth

    def plot_reward(self, title=None, xlabel="Episode", ylabel="Reward", figsize=(10, 6), save_path=None):
        """Plot reward metrics for all algorithms in the composition

        Args:
            title: Optional plot title
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure instead of displaying

        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        if not self.data:
            raise ValueError("No data loaded. Call get() first.")

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        for algo_name, algo_data in self.data.items():
            if 'smoothed_rewards' not in algo_data:
                continue

            color = self.colors.get(algo_name, "green")
            weight = self.composition.get(algo_name, 1.0)
            label = f"{algo_name.capitalize()} (w={weight:.2f})"

            # Plot the smoothed reward line
            ax.plot(
                algo_data['episodes_smooth'],
                algo_data['smoothed_rewards'],
                label=label,
                color=color
            )

            # Add the shaded area for standard deviation
            ax.fill_between(
                algo_data['episodes_smooth'],
                algo_data['smoothed_rewards'] - algo_data['std'],
                algo_data['smoothed_rewards'] + algo_data['std'],
                alpha=0.2,
                color=color
            )

        # Create a simplified title if none provided
        if title is None:
            if self.experiment:
                # Format experiment name properly (capitalize each word)
                env_name = ' '.join(word.capitalize()
                                    for word in self.experiment.split('_'))
                title = f"Reward - {env_name}"
            else:
                # Extract environment name from filename if possible
                filename = os.path.basename(
                    self.loaded_files[0]) if self.loaded_files else ""
                parts = filename.split('_')
                if len(parts) > 0:
                    env_name = parts[0].capitalize()
                    title = f"Reward - {env_name}"
                else:
                    title = "Reward Performance"

        # Set plot labels and styling
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Add a light background grid
        ax.set_axisbelow(True)

        # Set y-axis to start from zero if all rewards are positive
        if all(np.min(algo_data['smoothed_rewards']) >= 0 for algo_data in self.data.values()):
            ax.set_ylim(bottom=0)

        plt.tight_layout()

        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
