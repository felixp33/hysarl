# HySARL: Hybrid Simulation Architecture for Reinforcement Learning

![HySARL](https://img.shields.io/badge/HySARL-Reinforcement_Learning-blue)
![Python](https://img.shields.io/badge/Python-3.10-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

HySARL is a framework for reinforcement learning research that allows seamless integration and comparison of different physics simulation engines. The architecture supports training agents using multiple simulation backends while maintaining consistent interfaces and environments.

## 🔍 Overview

HySARL (Hybrid Simulation Architecture for Reinforcement Learning) addresses a common challenge in RL research: inconsistencies between simulation environments. By providing a unified framework that supports multiple physics engines (currently MuJoCo and Brax), HySARL enables:

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

HySARL consists of several key components:

1. **Environment Orchestrator**: Manages multiple simulation engines and provides a unified interface
2. **Composition Buffer**: Sophisticated replay buffer that supports stratified sampling from different sources
3. **Agents**: Implementation of SAC and TD3 algorithms
4. **Training Pipeline**: Coordinates training across multiple environments
5. **Metrics Collector**: Gathers and analyzes performance metrics

📊 Experiments
The repository contains several pre-configured experiments

