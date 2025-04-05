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

    def get(self, agent, experiment, composition=None, metric="rewards", data_dir="."):
        """Load metrics data for all files matching a specific experiment configuration

        Args:
            agent: Agent name (e.g., 'td3', 'sac')
            experiment: Environment name (e.g., 'halfcheetah', 'walker2d')
            composition: Dict mapping algorithm names to weights (e.g., {'mujoco': 1.0, 'brax': 0.0})
                        If None, defaults to {'mujoco': 1.0, 'brax': 0.0}
            metric: Metric to track (e.g., 'rewards', 'losses'). Defaults to 'rewards'
            data_dir: Directory to search for CSV files (default is current directory)

        Returns:
            self, for method chaining
        """
        self.agent = agent
        self.experiment = experiment.lower()
        self.metric = metric
        self.loaded_files = []

        # Set default composition if not provided
        if composition is None:
            composition = {'mujoco': 1.0, 'brax': 0.0}
        self.composition = composition

        # Calculate weights for pattern
        mujoco_weight = int(composition.get('mujoco', 0) * 100)
        brax_weight = int(composition.get('brax', 0) * 100)

        # Create the pattern to search for in filenames
        pattern = f"{agent}_{experiment}_m{mujoco_weight}b{brax_weight}_{metric}"

        # List all files in the directory and filter for CSV files containing our pattern
        all_files = os.listdir(data_dir)
        matching_files = [
            os.path.join(data_dir, f)
            for f in all_files
            if f.endswith('.csv') and pattern in f
        ]

        if not matching_files:
            raise FileNotFoundError(
                f"No files found matching pattern: {pattern}")

        # Track the number of files
        self.num_files = len(matching_files)
        print(f"Found {self.num_files} matching files for {pattern}")

        # Print the found files for debugging
        for file_path in matching_files:
            print(f"  - {os.path.basename(file_path)}")
            self.loaded_files.append(file_path)

        # Initialize the data dictionary
        self.data = {}

        # Initialize data collectors for averaging across files
        all_file_data = {algo: [] for algo in self.composition.keys()}

        # Load all matching files
        for file_path in matching_files:
            # Load file data
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

    def plot_reward(self, title=None, xlabel="Episode", ylabel="Reward", figsize=(10, 6), save_path=None, show=True):
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

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_single_engine(self, engine, window_size=10, title=None, xlabel="Episode", ylabel="Reward", figsize=(10, 6), save_path=None, show=True):
        """Plot reward metrics for a single engine, showing each individual run as a separate line

        Args:
            engine: Engine name to plot (e.g., 'mujoco', 'brax')
            window_size: Size of the moving average window (default: 10 episodes)
            title: Optional plot title
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure instead of displaying
            show: Whether to show the plot with plt.show() (default True)

        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        if not self.loaded_files:
            raise ValueError(
                "No data loaded. Call get() or get_by_filename() first.")

        if engine not in self.composition:
            raise ValueError(
                f"Engine '{engine}' not found in the composition. Available engines: {list(self.composition.keys())}")

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Load each file's data individually
        line_data = []
        for file_path in self.loaded_files:
            try:
                # Load CSV data
                df = pd.read_csv(file_path)

                # Check for required columns
                column_name = f"{engine}_reward"
                if column_name not in df.columns or 'episode' not in df.columns:
                    print(
                        f"Warning: File {os.path.basename(file_path)} missing required columns for {engine}")
                    continue

                # Extract data from columns
                episodes = df['episode'].values
                rewards = df[column_name].values

                # Apply smoothing with specified window size but preserve all timesteps
                if len(rewards) > 1:
                    # Create a smoothed version with same length as original
                    smoothed_rewards = np.zeros_like(rewards, dtype=float)

                    # For each point, take average of surrounding points within window
                    for i in range(len(rewards)):
                        # Calculate window bounds, handle edges properly
                        window_start = max(0, i - window_size // 2)
                        window_end = min(len(rewards), i +
                                         window_size // 2 + 1)
                        # Calculate mean for this window
                        smoothed_rewards[i] = np.mean(
                            rewards[window_start:window_end])

                    episodes_smooth = episodes  # Keep all episodes
                else:
                    smoothed_rewards = rewards
                    episodes_smooth = episodes

                # Add to our collection
                line_data.append({
                    'episodes': episodes_smooth,
                    'rewards': smoothed_rewards,
                    'file': os.path.basename(file_path)
                })

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

        if not line_data:
            raise ValueError(
                f"No valid data for engine '{engine}' in the loaded files")

        # Plot each file as a separate line
        color = self.colors.get(engine, "blue")
        color_variants = plt.cm.Blues(np.linspace(0.4, 0.8, len(line_data))) if engine == "mujoco" else \
            plt.cm.Oranges(np.linspace(0.4, 0.8, len(line_data)))

        # Plot individual lines
        for i, data in enumerate(line_data):
            # Get shortened filename by removing prefix and timestamp
            filename = data['file']
            # Extract just the timestamp part for label
            timestamp = filename.split('_')[-1].split('.')[0]

            # Plot with a variant of the base color
            ax.plot(
                data['episodes'],
                data['rewards'],
                label=f"Run {i+1} ({timestamp})",
                color=color_variants[i],
                alpha=0.8,
                linewidth=1.5
            )

        # Plot the mean as a thicker line if we have multiple files
        if len(line_data) > 1:
            # Find the minimum length to truncate all arrays (if we have multiple files)
            if len(line_data) > 1:
                # We're now using the full episode range for each file, so just use episodes directly
                min_length = min(len(data['episodes']) for data in line_data)
                episodes = line_data[0]['episodes'][:min_length]
                rewards_matrix = np.array(
                    [data['rewards'][:min_length] for data in line_data])
            # Calculate mean
            mean_rewards = np.mean(rewards_matrix, axis=0)

            # Plot mean with a thicker, more prominent line
            ax.plot(
                episodes,
                mean_rewards,
                label=f"{engine.capitalize()} Mean (n={len(line_data)})",
                color=color,
                linewidth=2.5
            )

        # Create title if none provided
        if title is None:
            weight = self.composition.get(engine, 1.0)
            if self.experiment:
                env_name = self.experiment.capitalize()
                title = f"{engine.capitalize()} Rewards (w={weight:.2f}) - {env_name}"
            else:
                title = f"{engine.capitalize()} Rewards (w={weight:.2f})"

        # Set plot labels and styling
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

        # Set y-axis to start from zero if all rewards are positive
        if all(np.min(data['rewards']) >= 0 for data in line_data):
            ax.set_ylim(bottom=0)

        plt.tight_layout()

        # Save if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # Show the plot only if requested
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_td_errors(self, agent, experiment, compositions, data_dir=".", window_size=10, title=None, xlabel="Time Step", ylabel="TD Error", figsize=(10, 6), save_path=None, show=True):
        """Plot TD errors for multiple compositions on a single plot

        Args:
            agent: Agent name (e.g., 'td3', 'sac')
            experiment: Environment name (e.g., 'halfcheetah', 'ant')
            compositions: List of composition dictionaries to compare
                        [{'mujoco': 1.0, 'brax': 0.0}, {'mujoco': 0.0, 'brax': 1.0}, ...]
            data_dir: Directory to search for CSV files
            window_size: Size of the moving average window
            title: Optional plot title
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure instead of displaying
            show: Whether to show the plot with plt.show() (default True)

        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Define a color cycle for different compositions
        color_cycle = plt.cm.tab10(np.linspace(0, 1, len(compositions)))

        # Track all loaded data for legend creation
        loaded_data = []

        # Process each composition
        for i, composition in enumerate(compositions):
            # Calculate weights for pattern
            mujoco_weight = int(composition.get('mujoco', 0) * 100)
            brax_weight = int(composition.get('brax', 0) * 100)

            # Create a label for this composition
            comp_label = f"m{mujoco_weight}b{brax_weight}"

            # Create the pattern to search for in filenames
            pattern = f"{agent}_{experiment}_m{mujoco_weight}b{brax_weight}_td_error"

            # Get all CSV files in directory
            all_files = os.listdir(data_dir)
            matching_files = [
                os.path.join(data_dir, f)
                for f in all_files
                if f.endswith('.csv') and pattern in f
            ]

            if not matching_files:
                print(f"Warning: No files found matching pattern: {pattern}")
                continue

            num_files = len(matching_files)
            print(f"Found {num_files} matching files for {pattern}")

            # Initialize data collector for this composition
            td_error_data = []

            # Load data from all matching files
            for file_path in matching_files:
                try:
                    # Load CSV data
                    df = pd.read_csv(file_path)

                    # Check columns
                    if 'td_error' not in df.columns:
                        print(
                            f"Warning: No 'td_error' column found in {os.path.basename(file_path)}")
                        continue

                    # Special handling for the fixed-episode format
                    if 'episode' in df.columns and df['episode'].nunique() <= 1:
                        # Use row index as episode number if episode column contains all the same value
                        episodes = np.arange(len(df))
                    elif 'episode' in df.columns:
                        episodes = df['episode'].values
                    else:
                        # If no episode column, use row index
                        episodes = np.arange(len(df))

                    # Extract TD error data
                    td_errors = df['td_error'].values

                    # Add to our data collection
                    td_error_data.append({
                        'episodes': episodes,
                        'td_errors': td_errors
                    })

                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

            # Process data for this composition
            if not td_error_data:
                continue

            # Find minimum episode length across all files
            min_length = min(len(data['episodes']) for data in td_error_data)

            # Create arrays for averaging
            all_episodes = td_error_data[0]['episodes'][:min_length]
            all_td_errors = np.array(
                [data['td_errors'][:min_length] for data in td_error_data])

            # Calculate mean TD error across files
            mean_td_errors = np.mean(all_td_errors, axis=0)

            # Apply smoothing while preserving all timesteps
            smoothed_td_errors = np.zeros_like(mean_td_errors, dtype=float)
            for j in range(len(mean_td_errors)):
                window_start = max(0, j - window_size // 2)
                window_end = min(len(mean_td_errors), j + window_size // 2 + 1)
                smoothed_td_errors[j] = np.mean(
                    mean_td_errors[window_start:window_end])

            # Plot the line with appropriate color
            line, = ax.plot(
                all_episodes,
                smoothed_td_errors,
                label=f"{comp_label} (n={num_files})",
                color=color_cycle[i],
                linewidth=2
            )

            # Add to loaded data for reference
            loaded_data.append({
                'composition': comp_label,
                'num_files': num_files,
                'line': line
            })

        # Check if we have data to plot
        if not loaded_data:
            raise ValueError(
                "No TD error data found for any of the specified compositions")

        # Create title if none provided
        if title is None:
            title = f"{agent.upper()} TD Error - {experiment.capitalize()}"

        # Set plot labels and styling
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

        plt.tight_layout()

        # Save if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # Show the plot only if requested
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig
