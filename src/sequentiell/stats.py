import time


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
        print("new TD errors len : ", len(td_errors))
        """Update TD errors for each training step"""
        self.td_errors.append(td_errors)
        print("Length of TD errors: ", len(self.td_errors))
        print("TD errors total: ", sum(len(l) for l in self.td_errors))
