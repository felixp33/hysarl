# HySaRL: Hybrid Sampling for Reinforcement Learning

![HySaRL](https://img.shields.io/badge/HySARL-Reinforcement_Learning-blue)
![Python](https://img.shields.io/badge/Python-3.10-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

HySARL is a framework for reinforcement learning research that allows seamless integration and comparison of different physics simulation engines. The architecture supports training agents using multiple simulation backends while maintaining consistent interfaces and environments.

## 🔍 Overview

HySaRL (Hybrid Simulation Architecture for Reinforcement Learning) addresses a common challenge in RL research: inconsistencies between simulation environments. By providing a unified framework that supports multiple physics engines (currently MuJoCo and Brax), HySaRL enables:

- Training on multiple simulation backends simultaneously
- Comparing performance across different physics engines
- Adaptive composition of replay buffers using experiences from multiple sources
- Dynamic engine dropout during training

## 🚀 Features

- **Multiple Physics Engines**: Seamless integration of MuJoCo and Brax physics engines
- **Unified Environment Interface**: Common interface for environments across different engines
- **Composition Buffer**: Advanced replay buffer that supports stratified sampling from different simulation sources
- **State-of-the-Art Algorithms**: Implementation of SAC (Soft Actor-Critic) and TD3 (Twin Delayed DDPG)
- **Visualization Tools**: Rich visualization and monitoring of training progress
- **Automated Evaluation**: Pipeline for systematic comparison between simulation environments

## 📋 Requirements

HySARL requires the following packages:
python=3.10
pytorch=2.2.2
gymnasium
brax
matplotlib
numpy
pandas
h5py

A full list of dependencies can be found in the `environment.yml` file.

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/felixp33/hysarl.git
cd hysarl

# Create and activate conda environment
conda env create -f environment.yml
conda activate hysarl
```

🏃‍♂️ Getting Started
Training a basic agent
```
pythonCopyfrom src.experiments import halfchetah_experiment_sac
from src.registration import register_all_envs

# Register all environments
register_all_envs()

# Run experiment with 50-50 composition between MuJoCo and Brax
halfchetah_experiment_sac({'mujoco': 0.5, 'brax': 0.5})
```

Visualizing results
```
import matplotlib.pyplot as plt
from src.viz import read_training_stats_h5, plot_training_stats

# Read training statistics from an H5 file
stats_data = read_training_stats_h5("logs/halfcheetah_20250325_204445.h5")

# Plot the statistics
fig = plot_training_stats(stats_data)
plt.show()

```

## 🏛️ Architecture

1. **Composition Buffer**: Sophisticated replay buffer that supports stratified sampling from different sources
2. **Environment Orchestrator**: Manages multiple simulation engines [MuJoCo, Brax] and provides a unified interface 
3. **Agents**: Implementation of SAC and TD3 algorithms
4. **Training Pipeline**: Coordinates training across multiple environments
5. **Metrics/Visualisation**: components to collect, read and anlayze data, including live dashboard

📊 Experiments
The repository contains several pre-configured experiments in the experiments





## 💻 Advanced Usage

For more advanced configurations, you can customize your training setup:

```python
from src.compostion_buffer import CompositionReplayBuffer
from src.agents.sac_agent import SACAgent
from src.sequentiell.pipeline import TrainingPipeline

# Define engines to use
engines = {'mujoco': 1, 'brax': 1}

# Create a composition buffer with custom parameters
composition_buffer = CompositionReplayBuffer(
    capacity=500000,
    strategy='stratified',
    sampling_composition={'mujoco': 0.5, 'brax': 0.5},  # Equal sampling from both engines
    buffer_composition={'mujoco': 1.0, 'brax': 1.0},    # Equal buffer space allocation
    engine_counts=engines,
    recency_bias=3.0                                    # Bias toward recent experiences
)

# Initialize SAC agent with custom hyperparameters
sac_agent = SACAgent(
    state_dim=17,                 # State dimension for HalfCheetah
    action_dim=6,                 # Action dimension for HalfCheetah
    replay_buffer=composition_buffer,
    hidden_dim=512,               # Size of hidden layers
    lr=3e-4,                      # Learning rate
    gamma=0.99,                   # Discount factor
    tau=0.005,                    # Target network update rate
    target_entropy=-0.5*6,        # Target entropy for automatic temperature tuning
    grad_clip=5.0,                # Gradient clipping threshold
    warmup_steps=20000            # Random exploration steps
)

# Set up the training pipeline
pipeline = TrainingPipeline(
    env_name='HalfCheetah',
    batch_size=100,
    episodes=500,
    steps_per_episode=1000,
    agent=sac_agent,
    engine_dropout=False,         # Disable engine dropout during training
    dashboard_active=True,        # Enable real-time dashboard
    engines_dict=engines
)

# Start training
pipeline.run()

```
