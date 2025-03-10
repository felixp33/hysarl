import gymnasium as gym
import sys
from gymnasium.envs.registration import register


def register_pybullet_envs():
    """Register PyBullet environments with Gymnasium."""
    try:
        # Try to import pybullet
        import pybullet
        print("✅ PyBullet is installed")

        # Check if environments are already registered
        try:
            env = gym.make('HalfCheetahBulletEnv-v0')
            env.close()
            print("✅ PyBullet environments already registered.")
            return
        except:
            print("🔄 Registering PyBullet environments manually")

        # Prevent pybullet_envs from registering environments itself
        # We'll save the original module and restore it later
        original_gym = sys.modules.get('gym')
        sys.modules['gym'] = None

        # Now import pybullet_envs without triggering its registration
        import pybullet_envs.gym_locomotion_envs
        import pybullet_envs.gym_manipulator_envs

        # Restore gym module
        sys.modules['gym'] = original_gym

        # Register environments manually
        register(
            id='HalfCheetahBulletEnv-v0',
            entry_point='pybullet_envs.gym_locomotion_envs:HalfCheetahBulletEnv',
            max_episode_steps=1000,
        )

        register(
            id='AntBulletEnv-v0',
            entry_point='pybullet_envs.gym_locomotion_envs:AntBulletEnv',
            max_episode_steps=1000,
        )

        register(
            id='ReacherBulletEnv-v0',
            entry_point='pybullet_envs.gym_manipulator_envs:ReacherBulletEnv',
            max_episode_steps=150,
        )

        print("✅ PyBullet environments registered successfully.")

    except ImportError as e:
        print(f"❌ PyBullet or pybullet_envs is not installed properly: {e}")
        print("Installation instructions:")
        print("pip install pybullet")
        print("pip install -e git+https://github.com/bulletphysics/bullet3.git#egg=pybullet_envs")
