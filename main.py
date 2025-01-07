if __name__ == "__main__":
    import numpy as np
    from environment import ParallelEnvironments
    from agent import SACAgent
    from buffer import ReplayBuffer

    # Parameters
    env_name = 'CartPole-v1'
    num_envs = 2
    buffer_capacity = 10000
    batch_size = 64
    episodes = 10
    steps_per_episode = 200

    # Initialize environments
    envs = ParallelEnvironments(env_name, num_envs)

    # Get state and action dimensions
    state_dim = envs.reset()[0].shape[0]
    action_dim = 1  # Discrete actions for CartPole (0 or 1)

    # Initialize replay buffer and agent
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    agent = SACAgent(state_dim, action_dim, replay_buffer)

    # Training Loop
    for episode in range(episodes):
        states = envs.reset()
        episode_rewards = [0 for _ in range(num_envs)]

        for step in range(steps_per_episode):
            # Select actions for each environment
            actions = [agent.select_action(state) for state in states]
            actions = [int(np.clip(action, 0, 1))
                       for action in actions]  # Ensure valid actions

            # Take a step in each environment
            next_states, rewards, dones = envs.step(actions)

            # Store experiences and update rewards
            for i in range(num_envs):
                agent.store_experience(
                    states[i], actions[i], rewards[i], next_states[i], dones[i])
                episode_rewards[i] += rewards[i]

            # Train agent
            agent.train(batch_size)

            # Update states
            states = next_states

            # Handle done environments
            if all(dones):
                break

        print(f"Episode {episode + 1}/{episodes}, Rewards: {episode_rewards}")

    # Close environments
    envs.close()
