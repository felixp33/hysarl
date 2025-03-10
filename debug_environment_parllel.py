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


class EnvironmentOrchestrator:
    def __init__(self, env_name, engines):
        self.env_name = env_name
        self.engines = {}  # Store engine for each worker by index
        self.workers = []
        self.conns = []
        self.active_envs = []
        self.worker_pids = {}

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
                self.worker_pids[env_id] = worker.pid  # Track worker PID
                # Store engine type by index
                self.engines[env_id] = engine_type
                env_id += 1

        print(f"✅ Created {len(self.workers)} environments.")
        print(f"🔍 Active environments: {self.active_envs}")
        print(f"🔍 Engines: {self.engines}")

    def step(self, actions):
        """Sends actions to the workers and retrieves step results."""
        try:
            print(f"🔄 Orchestrator step with {len(actions)} actions")
            results = []
            for i, (conn, action) in enumerate(zip(self.conns, actions)):
                if self.active_envs[i] and action is not None:
                    print(
                        f"📤 Sending step command to worker {i} ({self.engines[i]})")
                    conn.send(('step', action))

                    # Increased timeout to ensure operations complete
                    if not conn.poll(timeout=30.0):
                        print(f"⚠️ Worker {i} timeout! Restarting...")
                        self.restart_worker(i)
                        continue

                    response = conn.recv()
                    if isinstance(response, tuple) and len(response) == 2 and response[0] == "error":
                        print(f"❌ Error from worker {i}: {response[1]}")
                        continue

                    next_state, reward, done, env_identifier = response

                    # Ensure next_state is a numpy array
                    if not isinstance(next_state, np.ndarray):
                        next_state = np.array(next_state, dtype=np.float32)

                    print(
                        f"📥 Received response from worker {i}: reward={reward:.2f}, done={done}")
                    results.append((next_state, reward, done, env_identifier))

                    if done:
                        print(f"✓ Marking worker {i} as inactive (done=True)")
                        self.active_envs[i] = False  # Mark for reset

            if not results:
                print("⚠️ No results from any worker!")
                return [], [], [], []

            print(f"✅ Step complete, collected {len(results)} results")
            next_states, rewards, dones, env_ids = zip(*results)
            return next_states, rewards, dones, env_ids

        except Exception as e:
            print(f"❌ Error in `step()`: {e}")
            traceback.print_exc()
            self.close()
            raise

    def reset_specific(self, indices):
        """Reset only the environments at the specified indices."""
        try:
            print(f"🔄 Resetting environments at indices: {indices}")
            reset_states = []
            for idx in indices:
                if idx < len(self.conns):
                    conn = self.conns[idx]
                    print(
                        f"📤 Sending reset command to worker {idx} ({self.engines.get(idx, 'unknown')})")
                    conn.send(('reset', None))

                    # Increased timeout to ensure operations complete
                    if not conn.poll(timeout=30.0):
                        print(
                            f"⚠️ Worker {idx} timeout during reset! Restarting...")
                        self.restart_worker(idx)
                        reset_states.append(
                            np.zeros(env_specs[self.env_name]['state_dim'], dtype=np.float32))
                        continue

                    state = conn.recv()

                    # Ensure state is a numpy array
                    if not isinstance(state, np.ndarray):
                        state = np.array(state, dtype=np.float32)

                    print(
                        f"📥 Received reset state from worker {idx}, shape: {state.shape}")
                    reset_states.append(state)
                    self.active_envs[idx] = True
                    print(f"✓ Marked worker {idx} as active")

            print(f"✅ Reset complete, collected {len(reset_states)} states")
            return reset_states
        except Exception as e:
            print(f"❌ Error in `reset_specific()`: {e}")
            traceback.print_exc()
            self.close()
            raise

    def reset(self):
        """Reset all environments."""
        print("🔄 Resetting all environments")
        indices = list(range(len(self.conns)))
        return self.reset_specific(indices)

    def restart_worker(self, idx):
        """Restart a worker process if it becomes unresponsive."""
        print(f"🔄 Restarting Worker {idx}...")

        # Terminate old process if still running
        if idx < len(self.workers) and self.workers[idx].is_alive():
            try:
                self.workers[idx].terminate()
                self.workers[idx].join(timeout=2.0)

                # If still alive, force kill
                if self.workers[idx].is_alive():
                    try:
                        os.kill(self.workers[idx].pid, signal.SIGKILL)
                    except:
                        pass
            except Exception as e:
                print(f"⚠️ Error terminating worker {idx}: {e}")

        # Create a new worker
        # Default to first engine if idx not found
        engine_type = self.engines.get(idx, list(self.engines.values())[0])
        parent_conn, child_conn = Pipe()
        worker = EnvironmentWorker(
            self.env_name, engine_type, child_conn, idx)
        worker.start()

        # Replace old worker
        if idx < len(self.workers):
            self.workers[idx] = worker
        else:
            self.workers.append(worker)

        if idx < len(self.conns):
            self.conns[idx] = parent_conn
        else:
            self.conns.append(parent_conn)

        if idx < len(self.active_envs):
            self.active_envs[idx] = True
        else:
            self.active_envs.append(True)

        self.worker_pids[idx] = worker.pid
        print(f"✅ Worker {idx} ({engine_type}) restarted successfully")

    def close(self):
        """Properly close all environments and workers."""
        print("👋 Closing all environments...")
        try:
            for i, conn in enumerate(self.conns):
                try:
                    print(f"📤 Sending close command to worker {i}")
                    if conn.poll():
                        _ = conn.recv()
                    conn.send(('close', None))
                except:
                    pass  # Ignore errors during close

            for i, worker in enumerate(self.workers):
                try:
                    print(f"⏳ Waiting for worker {i} to join...")
                    worker.join(timeout=1.0)
                    if worker.is_alive():
                        print(f"🔄 Terminating worker {i}...")
                        worker.terminate()
                        time.sleep(0.1)
                        if worker.is_alive():
                            print(f"⚠️ Force killing worker {i}...")
                            try:
                                os.kill(worker.pid, signal.SIGKILL)
                            except:
                                pass
                except:
                    pass

            for conn in self.conns:
                try:
                    conn.close()
                except:
                    pass

            print("✅ All environments closed.")

        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            traceback.print_exc()
            for worker in self.workers:
                if worker.is_alive():
                    try:
                        worker.terminate()
                    except:
                        pass
