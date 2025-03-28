
from src.agents.sac_agent import SACAgent
from src.sequentiell.pipeline import TrainingPipeline
from src.compostion_buffer import CompositionReplayBuffer
from src.environment_orchestrator import env_specs


def halfchetah_experiment(sampling_composition, n_runs=1):
    for _ in range(n_runs):

        engines = {'mujoco': 1, 'brax': 1}

        composition_buffer = CompositionReplayBuffer(
            capacity=500000,
            strategy='stratified',
            sampling_composition=sampling_composition,
            buffer_composition={'mujoco': 1.0, 'brax': 1.0},
            engine_counts=engines,
            recency_bias=3.0
        )

        sac_agent = SACAgent(
            state_dim=17,
            action_dim=6,
            replay_buffer=composition_buffer,
            hidden_dim=512,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            target_entropy=-0.5*6,
            grad_clip=5.0,
            warmup_steps=20000
        )

        pipeline = TrainingPipeline(
            env_name='HalfCheetah',
            batch_size=256,
            episodes=500,
            steps_per_episode=1000,
            agent=sac_agent,
            engine_dropout=False,
            dashboard_active=True,
            engines_dict=engines
        )

        pipeline.run()


def walker_experiment(sampling_composition, n_runs=1):
    for _ in range(n_runs):

        engines = {'mujoco': 1, 'brax': 1}

        composition_buffer = CompositionReplayBuffer(
            capacity=1000000,
            strategy='stratified',
            sampling_composition=sampling_composition,
            buffer_composition={'mujoco': 1.0, 'brax': 1.0},
            engine_counts=engines,
            recency_bias=3.0
        )

        sac_agent_walker = SACAgent(
            state_dim=17,
            action_dim=6,
            replay_buffer=composition_buffer,
            hidden_dim=512,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            target_entropy=-6,
            grad_clip=5.0,
            warmup_steps=20000
        )

        pipeline = TrainingPipeline(
            env_name='Walker2d',
            batch_size=256,
            episodes=2000,
            steps_per_episode=1000,
            agent=sac_agent_walker,
            engine_dropout=False,
            dashboard_active=True,
            engines_dict=engines
        )

        pipeline.run()
