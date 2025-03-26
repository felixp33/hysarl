
from src.agents.sac_agent import SACAgent
from src.sequentiell.pipeline import TrainingPipeline
from src.compostion_buffer import CompositionReplayBuffer
from src.environment_orchestrator import env_specs


def halfchetah_experiment(sampling_composition, n=5):
    for _ in range(n):
        env_name = 'HalfCheetah'

        action_dim = 6
        engines = {'mujoco': 1, 'brax': 1}

        composition_buffer = CompositionReplayBuffer(
            capacity=500000,
            strategy='stratified',
            sampling_composition=sampling_composition,
            buffer_composition={'mujoco': 0.5, 'brax': 0.5},
            engine_counts=engines,
            recency_bias=3.0
        )

        sac_agent = SACAgent(
            state_dim=17,
            action_dim=action_dim,
            replay_buffer=composition_buffer,
            hidden_dim=512,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            target_entropy=-0.5*action_dim,
            grad_clip=5.0,
            warmup_steps=20000
        )

        pipeline = TrainingPipeline(
            env_name=env_name,
            batch_size=256,
            episodes=500,
            steps_per_episode=1000,
            agent=sac_agent,
            engine_dropout=False,
            dashboard_active=False,
            engines_dict=engines
        )

        pipeline.run()
