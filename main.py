from src.experiments import halfchetah_experiment_sac, walker_experiment_sac,  walker_experiment_td3, halfcheetah_experiment_td3, ant_experiment_sac
from src.registration import register_all_envs

import numpy as np

register_all_envs()

if __name__ == "__main__":
    ant_experiment_sac({'mujoco': 1, 'brax': 0}, n_runs=1)
