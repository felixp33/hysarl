from src.registration import register_all_envs
from src.compostion_buffer import CompositionReplayBuffer
from src.agents.sac_agent import SACAgent
from src.sequentiell.pipeline import TrainingPipeline
from src.environment_orchestrator import env_specs

import numpy as np
import torch
import time

# Register all environment types
register_all_envs()

if __name__ == "__main__":
    # Environment setup for HalfCheetah with all three engines
    env_name = 'HalfCheetah'
    # Use one instance of each engine
    engines = {'mujoco': 1, 'brax': 1}

    # Fetch dimensions from environment specs
    state_dim = env_specs[env_name]['state_dim']
    action_dim = env_specs[env_name]['action_dim']

    # Initialize the composition-controlled replay buffer
    composition_buffer = CompositionReplayBuffer(
        capacity=500000,
        strategy='stratified',
        sampling_composition={'mujoco': 0.5, 'brax': 0.5},
        buffer_composition={'mujoco': 0.5, 'brax': 0.5},
        engine_counts=engines,
        recency_bias=3.0
    )

    # Initialize SAC agent with optimal parameters for HalfCheetah
    sac_agent = SACAgent(
        state_dim=env_specs[env_name]['state_dim'],
        action_dim=env_specs[env_name]['action_dim'],
        replay_buffer=composition_buffer,
        hidden_dim=512,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        target_entropy=-0.5*action_dim,
        grad_clip=5.0,
        warmup_steps=20000
    )

    # Initialize the sequential training pipeline
    pipeline = TrainingPipeline(
        env_name=env_name,
        engines_dict=engines,
        batch_size=256,
        episodes=500,
        steps_per_episode=1000,
        agent=sac_agent,
        engine_dropout=True,
        drop_out_limit=0.5
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
        except Exception as e:
            print(f"⚠️ Could not save model: {e}")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        elapsed_time = time.time() - start_time
        print(f"⏱️ Ran for {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
