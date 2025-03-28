from src.agents.sac_agent import SACAgent


def sac_agent_halfchetah(buffer):
    return SACAgent(
        state_dim=17,
        action_dim=6,
        replay_buffer=buffer,
        hidden_dim=512,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        target_entropy=-0.5*6,
        grad_clip=5.0,
        warmup_steps=20000
    )


def sac_agent_walker(buffer):
    return SACAgent(
        state_dim=17,
        action_dim=6,
        replay_buffer=buffer,
        hidden_dim=512,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        target_entropy=6,
        grad_clip=5.0,
        warmup_steps=20000
    )
