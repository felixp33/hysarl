import gymnasium as gym
from multiprocessing import Process, Pipe

# Dictionary to store state and action dimensions for different environments
env_specs = {
    'CartPole-v1': {'state_dim': 4, 'action_dim': 2},
    'LunarLander-v2': {'state_dim': 8, 'action_dim': 4},
    'HalfCheetah-v4': {'state_dim': 17, 'action_dim': 6},
    'Ant-v4': {'state_dim': 111, 'action_dim': 8},
    'Humanoid-v4': {'state_dim': 376, 'action_dim': 17},
    # Continuous action space
    'Pendulum-v1': {'state_dim': 3, 'action_dim': 1},
    'MountainCarContinuous-v0': {'state_dim': 2, 'action_dim': 1}
}


class EnvironmentWorker(Process):
    def __init__(self, env_name, engine, conn, env_id):
        super(EnvironmentWorker, self).__init__()
        self.env_name = env_name
        self.engine = engine
        self.conn = conn
        self.env_id = env_id

    def run(self):
        # Dynamically load engine
        if self.engine == 'gym':
            env = gym.make(self.env_name)
        elif self.engine == 'mujoco':
            env = gym.make(self.env_name)
        elif self.engine == 'box2d':
            env = gym.make(self.env_name)
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")

        state, _ = env.reset()
        while True:
            cmd, data = self.conn.recv()
            if cmd == 'step':
                next_state, reward, terminated, truncated, _ = env.step(data)
                done = terminated or truncated
                print(
                    f"Worker {self.env_id}: State={state}, Action={data}, Reward={reward}, Next State={next_state}, Done={done}")
                if done:
                    next_state, _ = env.reset()
                self.conn.send((next_state, reward, done, self.env_id))
            elif cmd == 'reset':
                state, _ = env.reset()
                self.conn.send(state)
            elif cmd == 'close':
                env.close()
                self.conn.close()
                break


class EnvironmentOrchestrator:
    def __init__(self, env_name, engines):
        self.env_name = env_name
        self.engines = engines
        self.workers = []
        self.conns = []

        for i, engine in enumerate(engines):
            parent_conn, child_conn = Pipe()
            worker = EnvironmentWorker(env_name, engine, child_conn, env_id=i)
            worker.start()
            self.workers.append(worker)
            self.conns.append(parent_conn)

    def step(self, actions):
        for conn, action in zip(self.conns, actions):
            conn.send(('step', action))
        results = [conn.recv() for conn in self.conns]
        states, rewards, dones, env_ids = zip(*results)
        print(
            f"Main: Actions={actions}, States={states}, Rewards={rewards}, Dones={dones}, Env_IDs={env_ids}")
        return states, rewards, dones, env_ids

    def reset(self):
        for conn in self.conns:
            conn.send(('reset', None))
        states = [conn.recv() for conn in self.conns]
        print(f"Main: Reset States={states}")
        return states

    def close(self):
        for conn in self.conns:
            conn.send(('close', None))
        for worker in self.workers:
            worker.join()
