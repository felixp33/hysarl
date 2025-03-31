from registration.brax_registration import register_brax_envs
import gymnasium as gym
from multiprocessing import Process, Pipe
import numpy as np
import traceback
from src.environment_orchestrator import get_env_specs
# Import Brax registration
register_brax_envs()  # Register Brax environments with Gymnasium


class EnvironmentWorker(Process):
    def __init__(self, env_name, engine, conn, env_id):
        super(EnvironmentWorker, self).__init__()
        self.env_name = env_name
        self.engine = engine
        self.conn = conn
        self.env_id = env_id
        self.daemon = True

    def run(self):
        print(f"🚀 Worker {self.env_id} ({self.engine}) starting...")

        try:
            env = gym.make(
                get_env_specs(self.env_name)['engines'][self.engine])
            print(f"✅ Worker {self.env_id} ({self.engine}) initialized.")

            state, _ = env.reset()

            # Ensure state is a numpy array (important for Brax)
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)

            print(
                f"🔍 Worker {self.env_id} ({self.engine}) initial state shape: {state.shape}")

            while True:

                cmd, data = self.conn.recv()

                if cmd == 'step':

                    next_state, reward, terminated, truncated, _ = env.step(
                        data)

                    # Ensure next_state is a numpy array (important for Brax)
                    if not isinstance(next_state, np.ndarray):
                        next_state = np.array(next_state, dtype=np.float32)

                    done = terminated or truncated
                    env_identifier = f"{self.engine}_{self.env_id}"

                    self.conn.send((next_state, reward, done, env_identifier))

                elif cmd == 'reset':

                    state, _ = env.reset()

                    # Ensure state is a numpy array (important for Brax)
                    if not isinstance(state, np.ndarray):
                        state = np.array(state, dtype=np.float32)

                    self.conn.send(state)

                elif cmd == 'close':
                    env.close()
                    print(f"👋 Worker {self.env_id} ({self.engine}) closing...")
                    self.conn.close()
                    break

                else:
                    print(
                        f"❓ Worker {self.env_id} ({self.engine}) received unknown command: {cmd}")

        except Exception as e:
            print(f"❌ Error in worker {self.env_id} ({self.engine}): {e}")
            traceback.print_exc()
            try:
                self.conn.send(("error", str(e)))
            except:
                pass
            self.conn.close()
            raise
