#!/bin/bash
#SBATCH --job-name=eval_vit_npz_google_norm
#SBATCH --account=rrg-mpederso
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=65536M
#SBATCH --time=00:30:00
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err

# === Environment ===
source ~/envs/MambaFormer/bin/activate

# Extract ImageNet val to local NVMe
echo "Extracting ImageNet val..."
tar xf /project/def-mpederso/dataset/imagenet_val.tar -C $SLURM_TMPDIR
echo "Val extraction done."

cd ~/project/ViT2MambaFormer
mkdir -p scripts/logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python tools/evaluate_vit_npz.py \
    --npz models/vit/ViT-L_16.npz \
    --data_dir $SLURM_TMPDIR/ImageNet \
    --batch_size 128 \
    --image_size 384
