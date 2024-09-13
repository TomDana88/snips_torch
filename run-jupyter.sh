#!/bin/bash

# setup conda environment
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=ddrm_mdt
echo "*** Activating conda environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

NUM_CPU=2
NUM_GPU=1
echo "*** Creating jupyter-lab environment with $NUM_CPU CPUs and $NUM_GPU GPUs ***"
srun -c $NUM_CPU --gres=gpu:$NUM_GPU -w newton1 --pty jupyter-lab.sh

