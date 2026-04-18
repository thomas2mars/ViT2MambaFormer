#!/bin/bash
#SBATCH --job-name=viz_attention
#SBATCH --account=rrg-mpederso
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/viz_attention_%j.out
#SBATCH --error=logs/viz_attention_%j.err

# === Usage ===
# sbatch scripts/slurm/tools/submit_visualize_attention.sh \
#     --checkpoint WK/logs/distill_WK_base_v1/checkpoint_latest.pt \
#     --output_dir tools/attention_viz/WK_v1 \
#     --num_samples 10 \
#     --seed 42

# === Environment ===
source ~/envs/MambaFormer/bin/activate

# === Extract ImageNet val only (much faster than full train) ===
echo "Extracting ImageNet val to local NVMe scratch..."
tar xf /project/def-mpederso/dataset/imagenet_val.tar -C $SLURM_TMPDIR
echo "Val extraction done."

# === Run ===
cd ~/project/ViT2MambaFormer
mkdir -p logs

python tools/visualize_attention.py \
    --data_dir $SLURM_TMPDIR/ImageNet \
    "$@"
