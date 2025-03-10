# registration/__init__.py
from .brax_registration import register_brax_envs
from .pybullet_registration import register_pybullet_envs


def register_all_envs():
    """Register all environment types with Gymnasium."""
    register_brax_envs()
    register_pybullet_envs()
