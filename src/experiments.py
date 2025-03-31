from src.agents.sac_agent import SACAgent
from src.agents.td3_agent import TD3Agent
from src.sequentiell.pipeline import TrainingPipeline
from src.compostion_buffer import CompositionReplayBuffer


def halfchetah_experiment_sac(sampling_composition, n_runs=1):
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
            batch_size=100,
            episodes=500,
            steps_per_episode=1000,
            agent=sac_agent,
            engine_dropout=False,
            dashboard_active=True,
            engines_dict=engines
        )

        pipeline.run()


def walker_experiment_sac(sampling_composition, n_runs=1):
    for _ in range(n_runs):

        engines = {'mujoco': 1, 'brax': 1}

        composition_buffer = CompositionReplayBuffer(
            capacity=500000,
            strategy='stratified',
            sampling_composition=sampling_composition,
            buffer_composition={'mujoco': 0.5, 'brax': 0.5},
            engine_counts=engines,
            recency_bias=3.0
        )

        sac_agent_walker = SACAgent(
            state_dim=17,
            action_dim=6,
            replay_buffer=composition_buffer,
            hidden_dim=256,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            target_entropy=-6,
            grad_clip=1.0,
            warmup_steps=10000
        )

        pipeline = TrainingPipeline(
            env_name='Walker2d',
            batch_size=512,
            episodes=2000,
            steps_per_episode=1000,
            agent=sac_agent_walker,
            engine_dropout=False,
            dashboard_active=True,
            engines_dict=engines
        )

        pipeline.run()


def ant_experiment_sac(sampling_composition, n_runs=1):
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

        sac_agent = SACAgent(
            state_dim=27,
            action_dim=8,
            replay_buffer=composition_buffer,
            hidden_dim=512,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            target_entropy=-0.5*8,
            grad_clip=5.0,
            warmup_steps=20000
        )

        pipeline = TrainingPipeline(
            env_name='Ant',
            batch_size=256,
            episodes=500,
            steps_per_episode=1000,
            agent=sac_agent,
            engine_dropout=False,
            dashboard_active=True,
            engines_dict=engines
        )

        pipeline.run()


def halfcheetah_experiment_td3(sampling_composition, n_runs=1):
    for _ in range(n_runs):

        engines = {'mujoco': 1, 'brax': 1}

        composition_buffer = CompositionReplayBuffer(
            capacity=500000,
            strategy='stratified',
            sampling_composition=sampling_composition,
            buffer_composition={'mujoco': 0.5, 'brax': 0.5},
            engine_counts=engines,
            recency_bias=1.0
        )

        sac_agent_walker = TD3Agent(state_dim=17,
                                    action_dim=6,
                                    replay_buffer=composition_buffer,
                                    hidden_dim=512,
                                    lr=3e-4,
                                    gamma=0.99,
                                    tau=0.005,
                                    policy_noise=0.2,
                                    noise_clip=0.5,
                                    policy_delay=2)

        pipeline = TrainingPipeline(
            env_name='HalfCheetah',
            batch_size=512,
            episodes=2000,
            steps_per_episode=1000,
            agent=sac_agent_walker,
            engine_dropout=False,
            dashboard_active=True,
            engines_dict=engines
        )

        pipeline.run()


def walker_experiment_td3(sampling_composition, n_runs=1):
    for _ in range(n_runs):

        engines = {'mujoco': 1, 'brax': 1}

        composition_buffer = CompositionReplayBuffer(
            capacity=500000,
            strategy='stratified',
            sampling_composition=sampling_composition,
            buffer_composition={'mujoco': 0.5, 'brax': 0.5},
            engine_counts=engines,
            recency_bias=1.0
        )

        sac_agent_walker = TD3Agent(state_dim=17,
                                    action_dim=6,
                                    replay_buffer=composition_buffer,
                                    hidden_dim=256,
                                    lr=3e-4,
                                    gamma=0.99,
                                    tau=0.005,
                                    policy_noise=0.2,
                                    noise_clip=0.5,    # TD3-specific parameter
                                    policy_delay=2)    # TD3-specific parameter

        pipeline = TrainingPipeline(
            env_name='Walker2d',
            batch_size=512,
            episodes=2000,
            steps_per_episode=1000,
            agent=sac_agent_walker,
            engine_dropout=False,
            dashboard_active=True,
            engines_dict=engines
        )

        pipeline.run()
