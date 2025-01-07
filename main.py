from pipeline import TrainingPipeline
from replay_buffer import ReplayBuffer
from agents.dqn_agent import DQNAgent

if __name__ == "__main__":
    # Parameters
    env_name = 'CartPole-v1'
    engines = ['gym', 'mujoco', 'box2d']  # List of engines to test
    buffer_capacity = 1000
    batch_size = 64
    episodes = 1000
    steps_per_episode = 500

    # Initialize replay buffer and agent
    state_dim = 4
    action_dim = 2
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    agent = DQNAgent(state_dim, action_dim, replay_buffer)

    # Create and run training pipeline
    pipeline = TrainingPipeline(
        env_name, engines, buffer_capacity, batch_size, episodes, steps_per_episode, agent)
    pipeline.run()
    pipeline.plot_rewards()
