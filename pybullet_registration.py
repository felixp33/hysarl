# pybullet_registration.py
import gymnasium as gym
import pybullet
import os
import inspect
from gymnasium.envs.registration import register


def register_pybullet_envs():
    """Register PyBullet environments with Gymnasium."""
    try:
        # First, try importing the environments module from pybullet
        import pybullet_envs.gym_locomotion_envs

        # Register the environments if they're not already registered
        try:
            # Check if environments are already registered
            env = gym.make('HalfCheetahBulletEnv-v0')
            env.close()
            print("✅ PyBullet environments already registered.")
        except (gym.error.NameNotFound, KeyError, AttributeError):
            # We need to register the environments
            print("🔄 Registering PyBullet environments...")

            # Register HalfCheetah
            register(
                id='HalfCheetahBulletEnv-v0',
                entry_point='pybullet_envs.gym_locomotion_envs:HalfCheetahBulletEnv',
                max_episode_steps=1000,
            )

            # Register Ant
            register(
                id='AntBulletEnv-v0',
                entry_point='pybullet_envs.gym_locomotion_envs:AntBulletEnv',
                max_episode_steps=1000,
            )

            # Register Hopper
            register(
                id='HopperBulletEnv-v0',
                entry_point='pybullet_envs.gym_locomotion_envs:HopperBulletEnv',
                max_episode_steps=1000,
            )

            # Register ReacherBullet
            register(
                id='ReacherBulletEnv-v0',
                entry_point='pybullet_envs.gym_manipulator_envs:ReacherBulletEnv',
                max_episode_steps=150,
            )

            print("✅ PyBullet environments registered successfully.")

    except ImportError:
        print(
            "⚠️ Could not import pybullet_envs. Make sure PyBullet is installed correctly.")
        print("⚠️ Trying to find pybullet_envs in site-packages...")

        # Try to locate the module
        try:
            pybullet_path = os.path.dirname(inspect.getfile(pybullet))
            pybullet_envs_path = os.path.join(
                os.path.dirname(pybullet_path), 'pybullet_envs')
            if os.path.exists(pybullet_envs_path):
                print(f"✅ Found pybullet_envs at {pybullet_envs_path}")
                print("⚠️ You may need to add this directory to your PYTHONPATH")
            else:
                print("❌ Could not find pybullet_envs directory")
        except:
            print("❌ Could not locate pybullet installation")
