import gymnasium as gym
import numpy as np
import time

# Import Brax registration
from brax_registration import register_brax_envs
register_brax_envs()  # Register Brax environments with Gymnasium

# Environment specifications
# Environment specifications for both MuJoCo and Brax
env_specs = {
    'HalfCheetah': {
        'state_dim': 17,
        'action_dim': 6,
        'engines': {
            'mujoco': 'HalfCheetah-v4',
            'brax': 'BraxHalfCheetah-v0'
        }
    },
    'Ant': {
        'state_dim': 27,
        'action_dim': 8,
        'engines': {
            'mujoco': 'Ant-v4',
            'brax': 'BraxAnt-v0'
        }
    },
    'Humanoid': {
        'state_dim': 376,
        'action_dim': 17,
        'engines': {
            'mujoco': 'Humanoid-v4',
            'brax': 'BraxHumanoid-v0'
        }
    },
    'Walker2D': {
        'state_dim': 17,
        'action_dim': 6,
        'engines': {
            'mujoco': 'Walker2d-v4',
            'brax': 'BraxWalker2d-v0'
        }
    },
    'Reacher': {
        'state_dim': 11,
        'action_dim': 2,
        'engines': {
            'mujoco': 'Reacher-v4',
            'brax': 'BraxReacher-v0'
        }
    },
    'Hopper': {
        'state_dim': 11,
        'action_dim': 3,
        'engines': {
            'mujoco': 'Hopper-v4',
            'brax': 'BraxHopper-v0'
        }
    }
}


class EnvironmentOrchestrator:
    """Sequential environment manager that runs one environment at a time"""

    def __init__(self, env_name, engines_dict):
        """
        Initialize environments for each engine type.

        Args:
            env_name: Name of the environment (e.g., 'HalfCheetah')
            engines_dict: Dictionary mapping engine names to counts
        """
        self.env_name = env_name
        self.engines_dict = engines_dict
        self.envs = {}
        self.active_envs = []

        # Create one environment for each engine type
        for engine_type in engines_dict.keys():
            env_id = env_specs[env_name]['engines'][engine_type]
            try:
                self.envs[engine_type] = gym.make(env_id)
                print(f"✅ Created {engine_type} environment: {env_id}")
            except Exception as e:
                print(f"❌ Error creating {engine_type} environment: {e}")

        # Create a list of all engines (repeating based on count)
        for engine_type, count in engines_dict.items():
            for i in range(count):
                self.active_envs.append(engine_type)

        print(
            f"✅ Created {len(self.envs)} environment types for sequential training.")

    def run_episode(self, engine_type, agent, steps_per_episode, episode_num, stats=None):
        """
        Run a single episode on one environment.

        Args:
            engine_type: Type of engine to use ('mujoco', 'brax', etc.)
            agent: The agent to use for action selection
            steps_per_episode: Maximum steps per episode
            episode_num: Current episode number (for replay buffer)
            stats: Optional TrainingStats object to record step times

        Returns:
            Tuple of (total_reward, steps, done_flag)
        """
        env = self.envs[engine_type]
        state, _ = env.reset()
        total_reward = 0
        total_steps = 0

        # Ensure state is a numpy array (important for Brax)
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        for step in range(steps_per_episode):
            # Record timing if stats object is provided
            if stats is not None:
                stats.start_instance_timing(engine_type)

            # Select action
            action = agent.select_action(state)

            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Ensure next_state is a numpy array (important for Brax)
            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state, dtype=np.float32)

            done = terminated or truncated

            # Record step timing if stats object is provided
            if stats is not None:
                stats.end_instance_timing(engine_type)

            # Store transition in replay buffer
            env_id = f"{engine_type}_0"  # Use consistent ID format
            agent.replay_buffer.push(
                state, action, reward, next_state, done,
                env_id, episode_num
            )

            # Update agent's step counter
            agent.total_steps += 1

            # Update state and reward
            state = next_state
            total_reward += reward
            total_steps += 1

            # Break if episode is done
            if done:
                break

        return total_reward, total_steps, done

    def get_engine_types(self):
        """Get the list of all active engine types (with repeats based on count)"""
        return self.active_envs

    def close(self):
        """Close all environments"""
        for engine_type, env in self.envs.items():
            try:
                env.close()
                print(f"✅ Closed {engine_type} environment")
            except Exception as e:
                print(f"❌ Error closing {engine_type} environment: {e}")
