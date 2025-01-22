from pipeline import TrainingPipeline
from replay_buffer import ReplayBuffer
from agents.dqn_agent import DQNAgent
from environment import env_specs
from agents.sac_agent import SACAgent

if __name__ == "__main__":
    # Parameters
    env_name = 'Pendulum-v1'  # CartPole-v1, Pendulum-v1, LunarLander-v2
    # List of engines to test, gym = classic conrol/simple
    engines = ['gym']
    buffer_capacity = 100000
    batch_size = 256
    episodes = 1000
    steps_per_episode = 200
    sampling_strategy = 'uniform'  # stratified, ...

    buffer_compositon_type = 'standard'  # standard, fixed, ...
    buffer_compositon = [0.4, 0.4, 0.2]  # only if buffer is fixed

    # Fetch dimensions from environment specs
    state_dim = env_specs[env_name]['state_dim']
    action_dim = env_specs[env_name]['action_dim']

    # Initialize replay buffer and agent
    replay_buffer = ReplayBuffer(
        capacity=buffer_capacity, strategy=sampling_strategy)
    agent = SACAgent(state_dim, action_dim, replay_buffer,
                     hidden_dim=64, lr=1e-4)

    # Create and run training pipeline
    pipeline = TrainingPipeline(
        env_name, engines, buffer_capacity, batch_size, episodes, steps_per_episode, agent)
    pipeline.run()
    pipeline.plot_rewards()
