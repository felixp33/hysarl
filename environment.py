import gymnasium as gym
from multiprocessing import Process, Pipe

# Dictionary to store state and action dimensions for different environments
env_specs = {
    'CartPole-v1': {'state_dim': 4, 'action_dim': 2, 'engines': {'gym': 'CartPole-v1'}},
    'LunarLander-v2': {'state_dim': 8, 'action_dim': 4},
    'HalfCheetah-v4': {'state_dim': 17, 'action_dim': 6},
    'Ant-v4': {'state_dim': 111, 'action_dim': 8},
    'Humanoid-v4': {'state_dim': 376, 'action_dim': 17},
    # Continuous action space
    'Pendulum-v1': {
        'state_dim': 3,
        'action_dim': 1,
        'engines': {
            'gym': 'Pendulum-v1',
            'mujoco': 'InvertedPendulum-v5',
            'pybullet': 'InvertedPendulum-v5'  # Updated PyBullet env name
        }
    },
    'MountainCarContinuous-v0': {'state_dim': 2, 'action_dim': 1, 'engines': {'gym': 'MountainCarContinuous-v0'}},
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

                if cmd == 'step':
                    # Take step
                    next_state, reward, terminated, truncated, _ = env.step(
                        data)
                    done = terminated or truncated

                    # Reset if done
                    if done:
                        next_state, _ = env.reset()

                    self.conn.send((next_state, reward, done, self.env_id))
                    state = next_state  # Update state

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
        self.active_envs = [True] * len(engines)

        # Initialize workers and connections
        for i, engine in enumerate(engines):
            parent_conn, child_conn = Pipe()
            worker = EnvironmentWorker(env_name, engine, child_conn, env_id=i)
            worker.start()
            self.workers.append(worker)
            self.conns.append(parent_conn)

    def step(self, actions):
        try:
            # Only step active environments
            results = []
            for i, (conn, action) in enumerate(zip(self.conns, actions)):
                if self.active_envs[i] and action is not None:
                    conn.send(('step', action))
                    result = conn.recv()
                    results.append(result)

                    # Update environment status based on done flag
                    _, _, done, _ = result
                    if done:
                        self.active_envs[i] = False

            if not results:  # Handle case where no environments are active
                return [], [], [], []

            # Unzip results
            states, rewards, dones, env_ids = zip(*results)
            return states, rewards, dones, env_ids

        except Exception as e:
            print(f"Error in environment step: {e}")
            self.close()
            raise

    def reset(self):
        try:
            # Reset active environment tracking
            self.active_envs = [True] * len(self.engines)

            # Reset all environments
            for conn in self.conns:
                conn.send(('reset', None))

            # Collect initial states
            states = [conn.recv() for conn in self.conns]
            return states

        except Exception as e:
            print(f"Error in environment reset: {e}")
            self.close()
            raise

    def close(self):
        try:
            # Close all workers
            for conn in self.conns:
                if conn.poll():  # Clear any pending messages
                    _ = conn.recv()
                conn.send(('close', None))

            # Join workers with timeout
            for worker in self.workers:
                worker.join(timeout=1.0)
                if worker.is_alive():
                    worker.terminate()

            # Close all connections
            for conn in self.conns:
                conn.close()

        except Exception as e:
            print(f"Error during environment cleanup: {e}")
            # Force terminate any remaining workers
            for worker in self.workers:
                if worker.is_alive():
                    worker.terminate()
