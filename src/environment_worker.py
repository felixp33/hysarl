import gymnasium as gym
from multiprocessing import Process, Pipe
import numpy as np
import time
import traceback
import os
import signal

# Import Brax registration
from registration.brax_registration import register_brax_envs
register_brax_envs()  # Register Brax environments with Gymnasium

# Environment specifications
env_specs = {
    'HalfCheetah': {
        'state_dim': 17,
        'action_dim': 6,
        'engines': {
            'mujoco': 'HalfCheetah-v4',
            'brax': 'BraxHalfCheetah-v0'
        }
    },
    'Ant': {
        'state_dim': 27,
        'action_dim': 8,
        'engines': {
            'mujoco': 'Ant-v4',
            'brax': 'BraxAnt-v0'
        }
    }
}


class EnvironmentWorker(Process):
    def __init__(self, env_name, engine, conn, env_id):
        super(EnvironmentWorker, self).__init__()
        self.env_name = env_name
        self.engine = engine
        self.conn = conn
        self.env_id = env_id
        self.daemon = True  # Make workers daemonic so they exit when main process exits

    def run(self):
        print(f"🚀 Worker {self.env_id} ({self.engine}) starting...")

        try:
            # Create environment
            env = gym.make(env_specs[self.env_name]['engines'][self.engine])
            print(f"✅ Worker {self.env_id} ({self.engine}) initialized.")

            state, _ = env.reset()

            # Ensure state is a numpy array (important for Brax)
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)

            print(
                f"🔍 Worker {self.env_id} ({self.engine}) initial state shape: {state.shape}")

            while True:
                # Very long timeout to ensure operations complete
                if not self.conn.poll(timeout=60.0):
                    print(
                        f"⏳ Worker {self.env_id} ({self.engine}) waiting for command...")
                    continue

                cmd, data = self.conn.recv()
                print(
                    f"📝 Worker {self.env_id} ({self.engine}) received command: {cmd}")

                if cmd == 'step':
                    print(
                        f"🔄 Worker {self.env_id} ({self.engine}) stepping with action shape: {np.array(data).shape}")
                    next_state, reward, terminated, truncated, _ = env.step(
                        data)

                    # Ensure next_state is a numpy array (important for Brax)
                    if not isinstance(next_state, np.ndarray):
                        next_state = np.array(next_state, dtype=np.float32)

                    done = terminated or truncated
                    env_identifier = f"{self.engine}_{self.env_id}"

                    print(
                        f"✓ Worker {self.env_id} ({self.engine}) step complete: reward={reward:.2f}, done={done}")
                    self.conn.send((next_state, reward, done, env_identifier))

                elif cmd == 'reset':
                    print(
                        f"🔄 Worker {self.env_id} ({self.engine}) resetting...")
                    state, _ = env.reset()

                    # Ensure state is a numpy array (important for Brax)
                    if not isinstance(state, np.ndarray):
                        state = np.array(state, dtype=np.float32)

                    print(
                        f"✓ Worker {self.env_id} ({self.engine}) reset complete, state shape: {state.shape}")
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
