#!/usr/bin/env bash
# run_experiments.sh
# Simplified script: train with Weights & Biases on, toggling RGB-D usage (in_channels=3 or 4)

PYTHON=python
SCRIPT=rgbd_train.py

# Common arguments (using defaults for anything omitted)
EXP_BASE="temporal_anticipation_transformer"
WANDB_PROJECT="cholec80"

# Define configurations: suffix, rgbd_flag, in_channels
# "suffix,rgbd_flag,in_channels"
CONFIGS=(
  "rgb,,3"
  "rgbd,--use_rgbd,4"
)

for cfg in "${CONFIGS[@]}"; do
  IFS=',' read -r suffix rgbd_flag channels <<< "$cfg"
  exp_name="${EXP_BASE}_${suffix}"
  echo "Running experiment: $exp_name"

  $PYTHON "$SCRIPT" \
    --exp_name "$exp_name" \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --in_channels "$channels" \
    $rgbd_flag
  
  echo "---------------------------------------"
done
