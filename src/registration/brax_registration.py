import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import time
import jax
from brax import envs


def register_brax_envs():
    """Register Brax environments with Gymnasium."""
    try:
        # Check if environments are already registered
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
                start_time = time.time()
                self.state = self._reset_jit(subkey)  # Use JIT-compiled reset
                reset_time = time.time() - start_time

                # Convert to numpy array and ensure float32
                obs = np.array(self.state.obs, dtype=np.float32)
                return obs, {}
            except Exception as e:
                print(f"❌ Error in Brax reset: {e}")
                # Return zeros as fallback
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        def step(self, action):
            """Take a step in the environment using JIT-compiled functions."""
            try:
                # Convert action to the right format if needed
                action = np.array(action, dtype=np.float32)

                # Use JIT-compiled step function for performance
                start_time = time.time()
                self.state = self._step_jit(self.state, action)
                step_time = time.time() - start_time

                # Extract information and explicitly convert to numpy
                obs = np.array(self.state.obs, dtype=np.float32)
                reward = float(self.state.reward)
                done = bool(self.state.done)

                return obs, reward, done, False, {}
            except Exception as e:
                print(f"❌ Error in Brax step: {e}")
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

    # Initialize JAX to avoid first-call slowness
    print("Initializing JAX...")
    key = jax.random.PRNGKey(0)
    _ = jax.random.normal(key, (1,))

    # Pre-compile common JAX operations
    print("Pre-compiling JAX operations...")
    dummy_env = envs.get_environment('halfcheetah')
    dummy_key = jax.random.PRNGKey(0)
    dummy_state = dummy_env.reset(dummy_key)
    dummy_action = np.zeros((dummy_env.action_size,), dtype=np.float32)

    # JIT-compile step and measure time
    print("Compiling step function...")
    jit_step = jax.jit(dummy_env.step)
    start = time.time()
    _ = jit_step(dummy_state, dummy_action)
    first_compile = time.time() - start
    print(f"First step compilation took {first_compile:.3f}s")

    # Test compiled step speed
    start = time.time()
    _ = jit_step(dummy_state, dummy_action)
    second_step = time.time() - start
    print(f"Second step took {second_step:.3f}s (should be much faster)")

    # Register the environments
    try:
        print("Registering Brax HalfCheetah...")
        register(
            id='BraxHalfCheetah-v0',
            entry_point=lambda: FastBraxEnv('halfcheetah'),
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"❌ Error registering BraxHalfCheetah-v0: {e}")

    try:
        print("Registering Brax Ant...")
        register(
            id='BraxAnt-v0',
            entry_point=lambda: FastBraxEnv('ant'),
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"❌ Error registering BraxAnt-v0: {e}")

    print("✅ Brax environments registration complete")

    try:
        print("Registering Brax Hopper...")
        register(
            id='BraxHopper-v0',
            entry_point=lambda: FastBraxEnv('hopper'),
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"❌ Error registering BraxHopper-v0: {e}")

    try:
        print("Registering Brax CartPole...")
        register(
            id='BraxCartPole-v0',
            entry_point=lambda: FastBraxEnv('cartpole'),
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"❌ Error registering BraxCartPole-v0: {e}")

    try:
        print("Registering Brax Pendulum...")
        register(
            id='BraxPendulum-v0',
            entry_point=lambda: FastBraxEnv('pendulum'),
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"❌ Error registering BraxPendulum-v0: {e}")

    print("✅ Brax environments registration complete")
