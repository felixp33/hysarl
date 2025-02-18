import numpy as np
import matplotlib.pyplot as plt
from environment import EnvironmentOrchestrator
from dashboard import Dashboard

import numpy as np
import matplotlib.pyplot as plt
from environment import EnvironmentOrchestrator
from dashboard import Dashboard
import time
from replay_buffer import ReplayBuffer  # if needed


def convert_env_id(env_id):
    """
    Given an environment id string like "gym_0", return the numeric index (0).
    """
    if isinstance(env_id, str) and '_' in env_id:
        try:
            return int(env_id.split('_')[1])
        except (IndexError, ValueError):
            pass
    return int(env_id)


class TrainingStats:
    def __init__(self, engines_dict):
        self.engines = []
        for engine, count in engines_dict.items():
            self.engines.extend([engine] * count)
        self.unique_engines = list(engines_dict.keys())
        self.engine_indices = {engine: [] for engine in self.unique_engines}
        for i, engine in enumerate(self.engines):
            self.engine_indices[engine].append(i)
        self.instance_rewards = {i: [] for i in range(len(self.engines))}
        self.type_rewards = {engine: [] for engine in self.unique_engines}

    def update_rewards(self, episode_rewards):
        for i, reward in enumerate(episode_rewards):
            self.instance_rewards[i].append(reward)
        for engine_type in self.unique_engines:
            indices = self.engine_indices[engine_type]
            type_reward = np.mean([episode_rewards[i] for i in indices])
            self.type_rewards[engine_type].append(type_reward)

    def get_stats(self):
        return {'instance': self.instance_rewards, 'type': self.type_rewards}


class TrainingPipeline:
    def __init__(self, env_name, engines_dict, buffer_capacity, batch_size, episodes, steps_per_episode, agent):
        self.env_name = env_name
        self.engines_dict = engines_dict
        self.total_envs = sum(engines_dict.values())
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode

        self.envs = EnvironmentOrchestrator(env_name, engines_dict)
        self.rewards_history = []
        self.agent = agent
        self.stats = TrainingStats(engines_dict)
        self.dashboard = Dashboard(
            self.total_envs,
            {'Environment': env_name,
             'Engines': engines_dict,
             'Buffer Capacity': buffer_capacity,
             'Batch Size': batch_size,
             'Episodes': episodes,
             'Steps per Episode': steps_per_episode}
        )

    def run(self):
        try:
            print("Number of environments:", self.total_envs)
            plt.ion()
            for episode in range(self.episodes):
                states = self.envs.reset()
                episode_rewards = [0 for _ in range(self.total_envs)]
                episode_dones = {i: False for i in range(self.total_envs)}
                active_envs = [True] * self.total_envs
                env_steps = [0] * self.total_envs
                global_step = 0

                while any(steps < self.steps_per_episode and active
                          for steps, active in zip(env_steps, active_envs)):
                    actions = [
                        self.agent.select_action(state)
                        if active and steps < self.steps_per_episode
                        else None
                        for state, active, steps in zip(states, active_envs, env_steps)
                    ]
                    try:
                        next_states, rewards, dones, env_ids = self.envs.step(
                            actions)
                    except ValueError as e:
                        if not any(active_envs):
                            break
                        raise

                    env_indices = [convert_env_id(eid) for eid in env_ids]

                    for i, env_idx in enumerate(env_indices):
                        if active_envs[env_idx] and env_steps[env_idx] < self.steps_per_episode:
                            self.agent.replay_buffer.push(
                                states[env_idx],
                                actions[env_idx],
                                rewards[i],
                                next_states[i],
                                dones[i],
                                env_ids[i],
                                episode
                            )
                            episode_rewards[env_idx] += rewards[i]
                            env_steps[env_idx] += 1
                            if dones[i]:
                                episode_dones[env_idx] = True
                                active_envs[env_idx] = False
                                temp_states = self.envs.reset()
                                states[env_idx] = temp_states[env_idx]

                    for i, next_state in zip(env_indices, next_states):
                        if active_envs[i] and env_steps[i] < self.steps_per_episode:
                            states[i] = next_state

                    self.agent.train(self.batch_size)
                    global_step += 1

                    if not any(steps < self.steps_per_episode and active
                               for steps, active in zip(env_steps, active_envs)):
                        break

                mean_reward = np.mean(episode_rewards)
                self.rewards_history.append(mean_reward)
                self.stats.update_rewards(episode_rewards)
                type_stats = self.stats.get_stats()['type']

                if episode % 10 == 0:
                    print(f"Episode {episode + 1}/{self.episodes}")
                    for engine_type, rewards in type_stats.items():
                        print(f"{engine_type} Mean Reward: {rewards[-1]:.3f}")
                    print(f"Steps: {env_steps}")

                self.dashboard.update(
                    self.rewards_history,
                    self.agent.replay_buffer,
                    episode,
                    episode_dones,
                    self.stats
                )
        except Exception as e:
            print(f"Error during training: {e}")
            raise
        finally:
            self.envs.close()
            self.dashboard.close()
