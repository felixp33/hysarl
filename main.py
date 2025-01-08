from pipeline import TrainingPipeline
from replay_buffer import ReplayBuffer
from agents.dqn_agent import DQNAgent
from environment import env_specs
from agents.sac_agent import SACAgent

if __name__ == "__main__":
    # Parameters
    env_name = 'Pendulum-v1'  # CartPole-v1, Pendulum-v1, LunarLander-v2
    engines = ['gym', 'mujoco', 'box2d']  # List of engines to test
    buffer_capacity = 1000
    batch_size = 64
    episodes = 500
    steps_per_episode = 200
    sampling_strategy = 'uniform'  # stratified, ...

    buffer_compositon_type = 'standard'  # standard, fixed, ...
    buffer_compositon = [0.4, 0.4, 0.2]

    # Fetch dimensions from environment specs
    state_dim = env_specs[env_name]['state_dim']
    action_dim = env_specs[env_name]['action_dim']

    # Initialize replay buffer and agent
    replay_buffer = ReplayBuffer(
        capacity=buffer_capacity, strategy=sampling_strategy)
    agent = SACAgent(state_dim, action_dim, replay_buffer)

    # Create and run training pipeline
    pipeline = TrainingPipeline(
        env_name, engines, buffer_capacity, batch_size, episodes, steps_per_episode, agent)
    pipeline.run()
    pipeline.plot_rewards()
