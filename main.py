import numpy as np
from pipeline import TrainingPipeline
from environment import env_specs
from agents.sac_agent import SACAgent
from compostion_buffer import CompositionReplayBuffer

if __name__ == "__main__":
    # Environment setup for HalfCheetah with MuJoCo and PyBullet
    env_name = 'HalfCheetah'

    # Use one instance of each engine
    engines = {'mujoco': 1, 'pybullet': 1}

    # Training parameters
    buffer_capacity = 1000000  # 1M capacity
    batch_size = 512
    episodes = 500
    steps_per_episode = 1000

    # Fetch dimensions from environment specs
    state_dim = env_specs[env_name]['state_dim']
    action_dim = env_specs[env_name]['action_dim']

    print(
        f"Running HalfCheetah experiment with state_dim={state_dim}, action_dim={action_dim}")

    # Define target compositions for buffer and sampling
    # Equal balance between engines (adjust as needed)
    # {'mujoco': 1, 'pybullet': 0}
    buffer_composition = {'mujoco': 1.0, 'pybullet': 0.5}
    sampling_composition = {'mujoco': 1.0, 'pybullet': 0.5}

    # Initialize the composition-controlled replay buffer
    composition_buffer = CompositionReplayBuffer(
        capacity=buffer_capacity,
        strategy='stratified',  # Use stratified sampling
        sampling_composition=sampling_composition,
        buffer_composition=buffer_composition,
        engine_counts=engines  # Provide engine counts for initialization
    )

    # Initialize SAC agent with optimal parameters for HalfCheetah
    sac_agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        replay_buffer=composition_buffer,
        hidden_dim=256,      # Larger network for complex control
        lr=3e-4,             # Standard learning rate for SAC
        gamma=0.99,          # Standard discount factor
        tau=0.01,           # Soft target update rate
        alpha=0.3,           # Initial temperature parameter
        target_entropy=-action_dim,  # Heuristic for continuous control
        grad_clip=1.0,       # Prevent exploding gradients
        warmup_steps=10000   # Exploration phase
    )

    # Initialize and run training pipeline
    pipeline = TrainingPipeline(
        env_name=env_name,
        engines_dict=engines,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        agent=sac_agent
    )

    # Run training
    print("Starting training with Multi-Engine SAC using CompositionReplayBuffer...")
    pipeline.run()

    # Try to plot results if the method exists
    try:
        pipeline.plot_rewards()
    except AttributeError:
        print("Training complete. Results available in dashboard.")
