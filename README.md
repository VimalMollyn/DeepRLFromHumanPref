# RL-Teacher

`rl-teacher` is an implementation of [*Deep Reinforcement Learning from Human Preferences*](https://arxiv.org/abs/1706.03741) [Christiano et al., 2017].

**Updated for 2025** with modern libraries:
- **PyTorch** (replacing TensorFlow)
- **Gymnasium** (replacing OpenAI Gym)
- **Stable-Baselines3** (replacing custom TRPO/PPO)
- **MuJoCo** native Python bindings (replacing mujoco-py)

The system allows you to teach a reinforcement learning agent novel behaviors, even when both:

1. The behavior does not have a pre-defined reward function
2. A human can recognize the desired behavior, but cannot demonstrate it

<p align="center">
<img src="https://user-images.githubusercontent.com/306655/28396526-d4ce6334-6cb0-11e7-825c-63a85c8ff533.gif" />
</p>

## What's in this repository?

- A [reward predictor](/rl_teacher/teach.py) that can be plugged into any agent, and learns to predict which actions the human teacher would approve of
- A [webapp](/human-feedback-api) that humans can use to give feedback, providing the data used to train the reward predictor
- Integration with [Gymnasium](https://gymnasium.farama.org/) MuJoCo environments

## Installation

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Quick Start

```bash
# Clone the repository
git clone https://github.com/anthropics/rl-teacher.git
cd rl-teacher

# Create virtual environment and install with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

For the human feedback webapp:
```bash
uv pip install -e ".[human-feedback]"
```

## Usage

### Baseline RL

Run baseline reinforcement learning directly from the hard-coded reward function (no human feedback):

```bash
python -m rl_teacher.teach -p rl -e Hopper-v5 -n base-rl
```

View training in TensorBoard:
```bash
tensorboard --logdir ~/tb/rl-teacher/
```

### Synthetic Labels

Use synthetic feedback from the environment's true reward to train the reward predictor:

```bash
python -m rl_teacher.teach -p synth -l 1400 -e Hopper-v5 -n syn-1400
```

### Human Labels

To train with real human feedback, run two processes:

#### 1. Set up the webapp

```bash
cd human-feedback-api
python manage.py migrate
python manage.py collectstatic --noinput
python manage.py runserver 0.0.0.0:8000
```

Navigate to http://127.0.0.1:8000/ to access the labeling interface.

#### 2. Set up video storage (optional, for human feedback)

For human feedback, you'll need to store video clips. You can use Google Cloud Storage:

```bash
export RL_TEACHER_GCS_BUCKET="gs://your-bucket-name"
gsutil mb $RL_TEACHER_GCS_BUCKET
gsutil defacl ch -u AllUsers:R $RL_TEACHER_GCS_BUCKET
```

#### 3. Run the agent

```bash
python -m rl_teacher.teach -p human --pretrain_labels 175 -e Reacher-v5 -n human-175
```

## Command Line Options

```
-e, --env_id          Environment ID (e.g., Hopper-v5, Walker2d-v5)
-p, --predictor       Predictor type: rl, synth, or human
-n, --name            Experiment name
-s, --seed            Random seed (default: 1)
-w, --workers         Number of parallel workers (default: 4)
-l, --n_labels        Total number of labels to collect
-L, --pretrain_labels Number of pretraining labels
-t, --num_timesteps   Total training timesteps (default: 5e6)
-i, --pretrain_iters  Predictor pretraining iterations (default: 10000)
-V, --no_videos       Disable video recording
```

## Supported Environments

All Gymnasium MuJoCo v5 environments are supported:

- `Hopper-v5`
- `Walker2d-v5`
- `HalfCheetah-v5`
- `Ant-v5`
- `Humanoid-v5`
- `Reacher-v5`
- `Swimmer-v5`
- `InvertedPendulum-v5`
- `InvertedDoublePendulum-v5`

Use the "Short" prefix for shorter episodes (e.g., `ShortHopper-v5`).

## Architecture

<p align="center">
<img src="https://blog.openai.com/content/images/2017/06/diagram-4.png" />
</p>

The system works by:
1. Collecting trajectory segments from the agent
2. Presenting pairs of segments to a human (or synthetic oracle) for comparison
3. Training a reward predictor on these comparisons
4. Using the predicted reward to train the RL agent with PPO

## Acknowledgments

Based on the original implementation by OpenAI. Updated for modern Python and ML libraries in 2025.

A huge thanks to Paul Christiano and Dario Amodei for the design of this system.
