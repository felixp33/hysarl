from pipeline import TrainingPipeline
from replay_buffer import ReplayBuffer
from agents.dqn_agent import DQNAgent

if __name__ == "__main__":
    # Parameters

    env_name = 'CartPole-v1'
    num_envs = 2
    buffer_capacity = 10000
    batch_size = 64
    episodes = 100
    steps_per_episode = 200

    # Initialize replay buffer and agent
    state_dim = 4  # CartPole state dimension
    action_dim = 2  # CartPole action dimension
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    agent = DQNAgent(state_dim, action_dim, replay_buffer)

    # Create and run training pipeline
    pipeline = TrainingPipeline(
        env_name, num_envs, buffer_capacity, batch_size, episodes, steps_per_episode, agent)
    pipeline.run()
    pipeline.plot_rewards()
