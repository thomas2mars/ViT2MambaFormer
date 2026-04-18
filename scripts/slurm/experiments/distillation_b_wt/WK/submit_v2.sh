#!/bin/bash
#SBATCH --job-name=distill_WK_base_v2_mambaformer
#SBATCH --account=rrg-josedolz
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=4
#SBATCH --mem=256G
#SBATCH --time=11:00:00
#SBATCH --output=distillation_b_wt/WK/logs/%x_%j.out
#SBATCH --error=distillation_b_wt/WK/logs/%x_%j.err

# === Environment ===
source ~/envs/MambaFormer/bin/activate

# === Performance Tuning for H100 + NCCL ===
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL        # Use NVLink for intra-node P2P
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Extract ImageNet train + val to local NVMe ($SLURM_TMPDIR)
echo "Extracting ImageNet train to local NVMe scratch..."
tar xf ~/datasets/imagenet.tar -C $SLURM_TMPDIR --checkpoint=500000
echo "Train extraction done."

echo "Extracting ImageNet val to local NVMe scratch..."
tar xf ~/datasets/imagenet_val.tar -C $SLURM_TMPDIR
echo "Val extraction done."

# === Launch with torchrun (4 GPUs) ===
cd ~/project/ViT2MambaFormer

mkdir -p distillation_b_wt/WK/logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set RESUME=1 before sbatch to resume: RESUME=1 sbatch submit_v2.sh
RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
    RESUME_FLAG="--resume"
    echo "Resuming from latest checkpoint"
fi

torchrun --nproc_per_node=4 \
    distillation_b_wt/WK/main.py \
    --config config_v2 \
    --data_dir $SLURM_TMPDIR/ImageNet \
    $RESUME_FLAG
