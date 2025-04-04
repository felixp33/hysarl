from src.experiments import halfchetah_experiment_sac, walker_experiment_sac,  walker_experiment_td3, halfcheetah_experiment_td3, ant_experiment_sac, quick_test_experiment
from src.registration import register_all_envs

register_all_envs()

if __name__ == "__main__":
    halfcheetah_experiment_td3({'mujoco': 0.25, 'brax': 0.75}, n_runs=2)
