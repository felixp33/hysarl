import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import time
import jax
from brax import envs


def register_brax_envs():
    """Register Brax environments with Gymnasium."""

    try:
        jax.devices('gpu')
        print("GPU found, using GPU")
    except RuntimeError:
        jax.config.update('jax_platform_name', 'cpu')
        print("No GPU found, using CPU")

    try:
        env = gym.make('BraxHalfCheetah-v0')
        env.close()
        print("✅ Brax environments already registered.")
        return
    except (gym.error.NameNotFound, KeyError, AttributeError):
        pass

    # Create a wrapper class to make Brax environments compatible with Gymnasium
    class FastBraxEnv(gym.Env):
        """
        A fast wrapper for Brax environments using JAX JIT compilation.
        This ensures optimal performance while maintaining Gymnasium compatibility.
        """

        def __init__(self, env_name):
            print(f"Initializing fast Brax environment: {env_name}")
            # Initialize JAX random key
            self.key = jax.random.PRNGKey(int(time.time()))

            # Create Brax environment
            try:
                self.env = envs.get_environment(env_name)

                # Get the first state to determine observation shape
                self.key, subkey = jax.random.split(self.key)
                state = self.env.reset(subkey)

                # Define spaces
                obs_size = state.obs.shape[-1]
                action_size = self.env.action_size

                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(obs_size,),
                    dtype=np.float32
                )

                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(action_size,),
                    dtype=np.float32
                )

                # Save initial state
                self.state = state

                # JIT-compile the step and reset functions for performance
                self._reset_jit = jax.jit(self.env.reset)
                self._step_jit = jax.jit(self.env.step)

                print(f"✅ Fast Brax environment initialized with JIT compilation")
                print(f"  Observation space: {self.observation_space}")
                print(f"  Action space: {self.action_space}")

            except Exception as e:
                print(f"❌ Error initializing Brax environment: {e}")
                import traceback
                traceback.print_exc()
                # Create dummy spaces anyway to avoid errors
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(17,),  # Default for HalfCheetah
                    dtype=np.float32
                )
                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(6,),  # Default for HalfCheetah
                    dtype=np.float32
                )
                raise

        def reset(self, seed=None, options=None):
            """Reset the environment and return the initial observation."""
            try:
                if seed is not None:
                    self.key = jax.random.PRNGKey(seed)

                self.key, subkey = jax.random.split(self.key)
                self.state = self._reset_jit(subkey)  # Use JIT-compiled reset

                obs = np.array(self.state.obs, dtype=np.float32)
                return obs, {}
            except Exception as e:
                print(f"Error in Brax reset: {e}")
                # Return zeros as fallback
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        def step(self, action):
            """Take a step in the environment using JIT-compiled functions."""
            try:
                action = np.array(action, dtype=np.float32)

                self.state = self._step_jit(self.state, action)

                obs = np.array(self.state.obs, dtype=np.float32)
                reward = float(self.state.reward)
                done = bool(self.state.done)

                return obs, reward, done, False, {}
            except Exception as e:
                print(f"Error in Brax step: {e}")
                import traceback
                traceback.print_exc()
                # Return zeros as fallback
                return (
                    np.zeros(self.observation_space.shape, dtype=np.float32),
                    0.0,
                    True,
                    False,
                    {}
                )

        def close(self):
            """Close the environment."""
            # Nothing special needed for Brax environments
            pass

    dummy_env = envs.get_environment('halfcheetah')
    dummy_key = jax.random.PRNGKey(0)
    dummy_state = dummy_env.reset(dummy_key)
    dummy_action = np.zeros((dummy_env.action_size,), dtype=np.float32)

    jit_step = jax.jit(dummy_env.step)
    start = time.time()
    _ = jit_step(dummy_state, dummy_action)
    first_compile = time.time() - start

    start = time.time()
    _ = jit_step(dummy_state, dummy_action)
    second_step = time.time() - start

    brax_env_mapping = {
        'halfcheetah': 'BraxHalfCheetah-v0',
        'ant': 'BraxAnt-v0',
        'humanoid': 'BraxHumanoid-v0',
        'walker2d': 'BraxWalker2d-v0',
        'hopper': 'BraxHopper-v0',
        'reacher': 'BraxReacher-v0',
    }

    for brax_name, gym_id in brax_env_mapping.items():
        try:
            print(f"Registering {gym_id}...")
            register(
                id=gym_id,
                entry_point=lambda env_name=brax_name: FastBraxEnv(env_name),
                max_episode_steps=1000,
            )
            print(f"✅ Successfully registered {gym_id}")
        except Exception as e:
            print(f"❌ Error registering {gym_id}: {e}")

    print("✅ Brax environments registration complete")
