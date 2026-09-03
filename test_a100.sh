#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=buyin
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=test_a100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


# srun --account=b53017 --partition=buyin --gres=gpu:a100:1 --cpus-per-task=4 --mem=300G --time=08:00:00 --pty bash


nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
