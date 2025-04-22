#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR/../"
export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR/../../RL-X"


python experiment.py \
    --algorithm.name="ppo.default" \
    --algorithm.total_timesteps=1000_000_000 \
    --algorithm.nr_steps=2048 \
    --algorithm.minibatch_size=8192 \
    --algorithm.nr_epochs=5 \
    --algorithm.start_learning_rate=0.0004 \
    --algorithm.end_learning_rate=0.0 \
    --algorithm.entropy_coef=0.0 \
    --algorithm.gae_lambda=0.9 \
    --algorithm.critic_coef=1.0 \
    --algorithm.max_grad_norm=5.0 \
    --algorithm.clip_range=0.1 \
    --algorithm.evaluation_frequency=-1 \
    --algorithm.save_latest_frequency=6815744    \
    --algorithm.determine_fastest_cpu_for_gpu=True \
    --algorithm.device="gpu" \
    --environment.name="unitree_go2" \
    --environment.nr_envs=16 \
    --environment.async_skip_percentage=0.0 \
    --environment.cycle_cpu_affinity=True \
    --environment.seed=0 \
    --environment.reward_type="imitation" \
    --environment.domain_randomization_action_delay_type="none" \
    --environment.domain_randomization_mujoco_model_type="none" \
    --environment.domain_randomization_control_type="none" \
    --environment.domain_randomization_perturbation_type="none" \
    --environment.observation_dropout_type="none" \
    --environment.observation_noise_type="none" \
    --environment.domain_randomization_perturbation_sampling_type="none" \
    --environment.domain_randomization_sampling_type="none" \
    --environment.initial_state_type="default" \
    --runner.mode="train" \
    --runner.track_console=True \
    --runner.track_tb=False \
    --runner.track_wandb=True \
    --runner.save_model=True \
    --runner.wandb_entity="todo" \
    --runner.project_name="todo" \
    --runner.exp_name="E0" \
    --runner.notes=""
