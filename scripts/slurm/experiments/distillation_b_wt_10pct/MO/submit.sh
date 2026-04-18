#!/bin/bash
#SBATCH --job-name=distill_MO_base_10pct
#SBATCH --account=rrg-josedolz
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=2
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=distillation_b_wt_10pct/MO/logs/%x_%j.out
#SBATCH --error=distillation_b_wt_10pct/MO/logs/%x_%j.err

# === Environment ===
source ~/envs/MambaFormer/bin/activate

# === Performance Tuning for H100 + NCCL ===
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL        # Use NVLink for intra-node P2P
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Extract ImageNet train + val to local NVMe ($SLURM_TMPDIR)
echo "Extracting ImageNet train to local NVMe scratch..."
tar xf /project/def-mpederso/dataset/imagenet/imagenet.tar -C $SLURM_TMPDIR --checkpoint=50000
echo "Train extraction done."

echo "Extracting ImageNet val to local NVMe scratch..."
tar xf /project/def-mpederso/dataset/imagenet_val.tar -C $SLURM_TMPDIR
echo "Val extraction done."

# === Launch with torchrun (2 GPUs) ===
cd ~/project/ViT2MambaFormer

mkdir -p distillation_b_wt_10pct/MO/logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set RESUME=1 before sbatch to resume: RESUME=1 sbatch submit.sh
RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
    RESUME_FLAG="--resume"
    echo "Resuming from latest checkpoint"
fi

torchrun --nproc_per_node=2 \
    distillation_b_wt_10pct/MO/main.py \
    --data_dir $SLURM_TMPDIR/ImageNet \
    $RESUME_FLAG
