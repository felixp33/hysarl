# registration/__init__.py
from .brax_registration import register_brax_envs


def register_all_envs():
    """Register all environment types with Gymnasium."""
    register_brax_envs()
