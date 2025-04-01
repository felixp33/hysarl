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
        self.num_files = 0

    def get(self, composition, experiment, data_dir="."):
        """Load metrics data based on environment name and composition

        Args:
            composition: Dict mapping algorithm names to weights (e.g., {'mujoco': 0.8, 'brax': 0.2})
            experiment: Environment name (e.g., 'halfcheetah', 'walker2d')
            data_dir: Directory to search for CSV files

        Returns:
            self, for method chaining
        """
        self.composition = composition
        self.experiment = experiment.lower()
        self.data = {}
        self.loaded_files = []

        # Calculate weights for filename pattern
        mujoco_weight = int(composition.get('mujoco', 0) * 100)
        brax_weight = int(composition.get('brax', 0) * 100)

        # Create the file pattern based on the schema
        # Format: {environment}_m{mujoco_weight}_b{brax_weight}_*.csv
        file_pattern = f"{self.experiment.lower()}_m{mujoco_weight}_b{brax_weight}_*.csv"
        search_path = os.path.join(data_dir, file_pattern)

        # Find matching files
        matching_files = glob.glob(search_path)

        if not matching_files:
            raise FileNotFoundError(
                f"No files found matching pattern: {search_path}")

        # Track the number of files
        self.num_files = len(matching_files)
        print(f"Found {self.num_files} matching files")

        # Initialize data collectors for averaging across files
        all_file_data = {
            'mujoco': [],
            'brax': []
        }

        # Load all matching files
        for file_path in matching_files:
            self.loaded_files.append(file_path)

            # Load file data into the collector
            file_data = self._load_file_data(file_path)

            # Add to collectors
            for algo in self.composition.keys():
                if algo in file_data:
                    all_file_data[algo].append(file_data[algo])

        # Process aggregate data for each algorithm
        for algo_name in self.composition.keys():
            algo_files = all_file_data[algo_name]

            if not algo_files:
                continue

            # Ensure all files have data available for this algorithm
            if len(algo_files) != self.num_files:
                print(
                    f"Warning: Only {len(algo_files)} of {self.num_files} files have data for {algo_name}")

            # Process aggregate data
            self._process_aggregate_data(algo_name, algo_files)

        return self

    def get_by_filename(self, composition, file_path):
        """Load metrics data directly from a specific file

        Args:
            composition: Dict mapping algorithm names to weights (e.g., {'mujoco': 0.8, 'brax': 0.2})
            file_path: Direct path to a CSV file

        Returns:
            self, for method chaining
        """
        self.composition = composition
        self.data = {}
        self.loaded_files = []

        # Extract experiment name from filename if possible
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) > 0:
            self.experiment = parts[0].lower()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.loaded_files.append(file_path)
        self.num_files = 1

        # Load file data
        file_data = self._load_file_data(file_path)

        # Process data for each algorithm
        for algo_name in self.composition.keys():
            if algo_name in file_data:
                # For a single file, just use the file data directly
                self._process_algorithm_data(
                    algo_name,
                    file_data[algo_name]['episodes'],
                    file_data[algo_name]['rewards']
                )

        return self

    def _load_file_data(self, file_path):
        """Load data from a CSV file and return it without processing

        Args:
            file_path: Path to the CSV file

        Returns:
            dict: Dictionary with algorithm data
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

            # Collect data for each algorithm
            file_data = {}

            if 'mujoco' in self.composition:
                mujoco_rewards = df['mujoco_reward'].values
                file_data['mujoco'] = {
                    'episodes': episodes,
                    'rewards': mujoco_rewards
                }

            if 'brax' in self.composition:
                brax_rewards = df['brax_reward'].values
                file_data['brax'] = {
                    'episodes': episodes,
                    'rewards': brax_rewards
                }

            return file_data

        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return {}

    def _process_aggregate_data(self, algo_name, file_data_list):
        """Process aggregate data across multiple files for an algorithm

        Args:
            algo_name: Name of the algorithm
            file_data_list: List of data dictionaries from multiple files
        """
        # First, ensure all files have the same episode lengths
        # If not, we'll need to interpolate to a common x-axis
        episode_arrays = [data['episodes'] for data in file_data_list]
        reward_arrays = [data['rewards'] for data in file_data_list]

        # Find the shortest episode length to use as reference
        min_episodes = min(len(episodes) for episodes in episode_arrays)

        # Truncate all arrays to the minimum length
        episodes = episode_arrays[0][:min_episodes]
        rewards_matrix = np.array([rewards[:min_episodes]
                                  for rewards in reward_arrays])

        # Calculate mean and std across files at each episode
        mean_rewards = np.mean(rewards_matrix, axis=0)

        # For std, we'll compute the combined standard deviation across files
        if len(file_data_list) > 1:
            std_rewards = np.std(rewards_matrix, axis=0)
        else:
            # For a single file, we'll estimate std using a rolling window
            std_rewards = np.zeros_like(mean_rewards)
            window = min(self.window_size, len(mean_rewards) // 5)
            if window > 1:
                for i in range(len(mean_rewards)):
                    start = max(0, i - window // 2)
                    end = min(len(mean_rewards), i + window // 2 + 1)
                    std_rewards[i] = np.std(mean_rewards[start:end])

        # Apply smoothing to mean rewards
        if len(mean_rewards) >= self.window_size:
            smoothed_rewards = sliding_window_avg(
                mean_rewards, self.window_size)

            # Also smooth the std to match
            smoothed_std = sliding_window_avg(std_rewards, self.window_size)

            # Adjusted episodes to match smoothed array length
            offset = (self.window_size - 1) // 2
            episodes_smooth = episodes[offset:-(offset)]
        else:
            # Not enough data points for smoothing, use raw data
            smoothed_rewards = mean_rewards
            smoothed_std = std_rewards
            episodes_smooth = episodes

        # Store the processed data
        self.data[algo_name] = {
            'episodes': episodes,
            'rewards': mean_rewards,
            'smoothed_rewards': smoothed_rewards,
            'std': smoothed_std,
            'episodes_smooth': episodes_smooth
        }

        # Set the episodes reference if not already set
        if self.timesteps is None and len(episodes_smooth) > 0:
            self.timesteps = episodes_smooth

    def _process_algorithm_data(self, algo_name, episodes, rewards):
        """Process data for a single algorithm from a single file"""
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
            raise ValueError(
                "No data loaded. Call get() or get_by_filename() first.")

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        for algo_name, algo_data in self.data.items():
            if 'smoothed_rewards' not in algo_data:
                continue

            color = self.colors.get(algo_name, "green")
            weight = self.composition.get(algo_name, 1.0)

            # Add seed count to label if multiple files
            if self.num_files > 1:
                label = f"{algo_name.capitalize()} (w={weight:.2f}, n={self.num_files})"
            else:
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
                # Format experiment name properly (capitalize first letter)
                env_name = self.experiment.capitalize()
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
