import gymnasium as gym
from multiprocessing import Process, Pipe

# Environment specifications
env_specs = {
    'CartPole-v1': {
        'state_dim': 4,
        'action_dim': 2,
        'engines': {
            'gym': 'CartPole-v1',
            'mujoco': 'CartPole-v1',
            'pybullet': 'CartPole-v1'
        }
    },
    'Pendulum-v1': {
        'state_dim': 3,
        'action_dim': 1,
        'engines': {
            'gym': 'Pendulum-v1',
            'mujoco': 'InvertedPendulum-v5',
            'pybullet': 'InvertedPendulum-v5'
        }
    },
    'MountainCarContinuous-v0': {
        'state_dim': 2,
        'action_dim': 1,
        'engines': {'gym': 'MountainCarContinuous-v0'}
    }
}


class EnvironmentWorker(Process):
    def __init__(self, env_name, engine, conn, env_id):
        super(EnvironmentWorker, self).__init__()
        self.env_name = env_name
        self.engine = engine
        self.conn = conn
        self.env_id = env_id

    def run(self):
        try:
            env = gym.make(env_specs[self.env_name]['engines'][self.engine])
            state, _ = env.reset()

            while True:
                cmd, data = self.conn.recv()
                print(f"Worker {self.env_id} received {cmd} with data {data}")

                if cmd == 'step':
                    next_state, reward, terminated, truncated, _ = env.step(
                        data)
                    done = terminated or truncated
                    if done:
                        next_state, _ = env.reset()
                    # Format the identifier as "engine_envid", e.g. "gym_0"
                    env_identifier = f"{self.engine}_{self.env_id}"
                    self.conn.send((next_state, reward, done, env_identifier))

                elif cmd == 'reset':
                    state, _ = env.reset()
                    self.conn.send(state)

                elif cmd == 'close':
                    env.close()
                    self.conn.close()
                    break

        except Exception as e:
            print(f"Error in worker {self.env_id}: {e}")
            self.conn.close()
            raise


class EnvironmentOrchestrator:
    def __init__(self, env_name, engines):
        self.env_name = env_name
        self.engines = engines
        self.workers = []
        self.conns = []
        self.active_envs = []

        env_id = 0
        for engine_type, count in engines.items():
            for _ in range(count):
                parent_conn, child_conn = Pipe()
                worker = EnvironmentWorker(
                    env_name, engine_type, child_conn, env_id)
                worker.start()
                self.workers.append(worker)
                self.conns.append(parent_conn)
                self.active_envs.append(True)
                env_id += 1

        print("Creating environments:", engines)

    def step(self, actions):
        try:
            results = []
            for i, (conn, action) in enumerate(zip(self.conns, actions)):
                if self.active_envs[i] and action is not None:
                    conn.send(('step', action))
                    if not conn.poll(timeout=1.0):
                        print(f"Timeout waiting for worker {i}")
                        continue
                    next_state, reward, done, env_identifier = conn.recv()
                    results.append((next_state, reward, done, env_identifier))
            return zip(*results) if results else ([], [], [], [])
        except Exception as e:
            print(f"Error in environment step: {e}")
            self.close()
            raise

    def reset(self):
        try:
            self.active_envs = [True] * len(self.conns)
            for conn in self.conns:
                conn.send(('reset', None))
            states = [conn.recv() for conn in self.conns]
            return states
        except Exception as e:
            print(f"Error in environment reset: {e}")
            self.close()
            raise

    def close(self):
        try:
            for conn in self.conns:
                if conn.poll():
                    _ = conn.recv()
                conn.send(('close', None))

            for worker in self.workers:
                worker.join(timeout=1.0)
                if worker.is_alive():
                    worker.terminate()

            for conn in self.conns:
                conn.close()
        except Exception as e:
            print(f"Error during environment cleanup: {e}")
            for worker in self.workers:
                if worker.is_alive():
                    worker.terminate()
