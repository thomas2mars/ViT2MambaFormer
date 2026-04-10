#!/bin/bash
#SBATCH --job-name=benchmark_base_fps
#SBATCH --account=rrg-mpederso
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=65536M
#SBATCH --time=00:30:00
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err

# === Environment ===
source /home/t2mars/envs/MambaFormer/bin/activate

cd /project/6007600/t2mars/dist_vision_mamba
mkdir -p scripts/logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python MambaFormer/benchmarks/benchmark_fps_base.py
