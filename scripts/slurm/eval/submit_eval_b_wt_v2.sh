#!/bin/bash
#SBATCH --job-name=eval_b_wt_v2
#SBATCH --account=rrg-josedolz
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=65536M
#SBATCH --time=00:15:00
#SBATCH --output=distillation_b_wt/WK/logs/%x_%j.out
#SBATCH --error=distillation_b_wt/WK/logs/%x_%j.err

# === Environment ===
source ~/envs/MambaFormer/bin/activate

# Extract ImageNet val to local NVMe
echo "Extracting ImageNet val..."
tar xf ~/datasets/imagenet_val.tar -C $SLURM_TMPDIR
echo "Val extraction done."

cd ~/project/ViT2MambaFormer
mkdir -p distillation_b_wt/WK/logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python tools/evaluate_mambaformer.py \
    --checkpoint distillation_b_wt/WK/logs/distill_WK_base_v2/checkpoint_latest.pt \
    --data_dir $SLURM_TMPDIR/ImageNet \
    --image_size 384