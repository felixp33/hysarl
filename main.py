from pipeline import TrainingPipeline
from replay_buffer import ReplayBuffer
from environment import env_specs
from agents.sac_agent import SACAgent
from agents.dqn_cont_agent import DQNAgentCont
from agents.dqn_disc_agent import DQNAgentDisc

if __name__ == "__main__":
    # CartPole-v1, Pendulum-v1, LunarLander-v2,MountainCarContinuous-v0
    env_name = 'CartPole-v1'
    # List of engines to test, gym = classic conrol/simple
    engines = engines = {'gym': 3, 'mujoco': 2}
    buffer_capacity = 10000
    batch_size = 256
    episodes = 1000
    steps_per_episode = 500
    sampling_strategies = ['uniform', 'stratified', 'stratified_shift']
    sampling_composition = {'gym': 0.3, 'mujoco': 0.7}
    buffer_compositon_type = ['uniform', 'stratified', '_']
    buffer_composition = {'gym': 0.3, 'mujoco': 0.7}

    # Fetch dimensions from environment specs
    state_dim = env_specs[env_name]['state_dim']
    action_dim = env_specs[env_name]['action_dim']

    # Initialize replay buffer and agent
    replay_buffer = ReplayBuffer(
        capacity=buffer_capacity, strategy=sampling_strategies[1], composition=sampling_composition)
    replay_buffer_shift = ReplayBuffer(
        capacity=buffer_capacity, strategy=sampling_strategies[2], composition=buffer_composition, start_composition={'gym': 0.8, 'mujoco': 0.2}, goal_composition={'gym': 0.2, 'mujoco': 0.8}, episode_goal_reached=200)
    sac_agent = SACAgent(state_dim, action_dim, replay_buffer,
                         hidden_dim=128, lr=1e-4, alpha=0.2, warmup_steps=10000)
    dqn_agent = DQNAgentDisc(state_dim, action_dim, replay_buffer_shift,
                             lr=1e-4, gamma=0.99, epsilon_decay=1e-3)
    # Create and run training pipeline
    pipeline = TrainingPipeline(
        env_name, engines, buffer_capacity, batch_size, episodes, steps_per_episode, dqn_agent)
    pipeline.run()
    pipeline.plot_rewards()
