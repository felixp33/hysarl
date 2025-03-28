from src.experiment import halfchetah_experiment, walker_experiment
from src.registration import register_all_envs
from src.compostion_buffer import CompositionReplayBuffer
from src.agents.sac_agent import SACAgent
from src.sequentiell.pipeline import TrainingPipeline
from src.environment_orchestrator import env_specs

import numpy as np
import torch
import time

# Register all environment types
register_all_envs()

if __name__ == "__main__":
    walker_experiment({'mujoco': 0.5, 'brax': 0.5}, n_runs=1)
