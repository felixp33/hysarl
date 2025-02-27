from compostion_buffer import CompositionReplayBuffer
from agents.sac_agent import SACAgent
import numpy as np
import torch
from sequentiell.pipeline import TrainingPipeline
from sequentiell.environment import env_specs
import time

# Register Brax environments
from brax_registration import register_brax_envs
register_brax_envs()

# Import agent and replay buffer

if __name__ == "__main__":
    # Environment setup for HalfCheetah with MuJoCo and Brax
    env_name = 'HalfCheetah'
    # Use one instance of each engine
    engines = {'mujoco': 1, 'brax': 1}

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
    buffer_composition = {'mujoco': 0.5, 'brax': 0.5}
    sampling_composition = {'mujoco': 0.5, 'brax': 0.5}

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
        tau=0.01,            # Soft target update rate
        alpha=0.3,           # Initial temperature parameter
        target_entropy=-action_dim,  # Heuristic for continuous control
        grad_clip=1.0,       # Prevent exploding gradients
        warmup_steps=10000   # Exploration phase
    )

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize the sequential training pipeline
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
    print("Starting training with Sequential Multi-Engine SAC...")
    start_time = time.time()

    try:
        pipeline.run()

        total_time = time.time() - start_time
        print(f"✅ Training completed in {total_time:.2f} seconds")

        # Save model
        try:
            save_path = f"sac_model_{env_name}.pt"
            sac_agent.save(save_path)
            print(f"✅ Model saved to {save_path}")
        except:
            print("⚠️ Could not save model")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        elapsed_time = time.time() - start_time
        print(f"⏱️ Ran for {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
