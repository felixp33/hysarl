import gymnasium as gym
from multiprocessing import Process, Pipe


class EnvironmentWorker(Process):
    def __init__(self, env_name, engine, conn):
        super(EnvironmentWorker, self).__init__()
        self.env_name = env_name
        self.engine = engine
        self.conn = conn

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
                    f"Worker: State={state}, Action={data}, Reward={reward}, Next State={next_state}, Done={done}")
                if done:
                    next_state, _ = env.reset()
                self.conn.send((next_state, reward, done))
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

        for engine in engines:
            parent_conn, child_conn = Pipe()
            worker = EnvironmentWorker(env_name, engine, child_conn)
            worker.start()
            self.workers.append(worker)
            self.conns.append(parent_conn)

    def step(self, actions):
        for conn, action in zip(self.conns, actions):
            conn.send(('step', action))
        results = [conn.recv() for conn in self.conns]
        states, rewards, dones = zip(*results)
        print(
            f"Main: Actions={actions}, States={states}, Rewards={rewards}, Dones={dones}")
        return states, rewards, dones

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
