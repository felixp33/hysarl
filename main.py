from src.experiments import halfchetah_experiment_sac, walker_experiment_sac,  walker_experiment_td3, halfcheetah_experiment_td3
from src.registration import register_all_envs

import numpy as np

register_all_envs()

if __name__ == "__main__":
    halfchetah_experiment_sac({'mujoco': 0.5, 'brax': 0.5}, n_runs=1)
