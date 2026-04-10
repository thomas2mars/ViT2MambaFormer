# Distillation Step MO — Attention Map Alignment

Distills a ViT-Large teacher into a MambaFormer student by aligning Mamba2 SSD attention maps to ViT self-attention maps, layer by layer.

## What is being trained

Only the Mamba2 `in_proj` weights (which produce B, C, w states for attention maps) and optional bottleneck `down_proj`/`up_proj` layers are trained. Everything else (LN, MLP, embeddings, head) is frozen and loaded from the ViT teacher weights.

- **Trainable**: ~101M / 333M total parameters
- **Loss**: Combined Frobenius norm (CLS rows, CLS cols, patches) + KL divergence
- **Metrics**: Cosine similarity and JS divergence between student/teacher attention maps

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters (LR, epochs, paths, etc.) |
| `main_multi_ddp.py` | Multi-GPU training script (DDP, 4x H100) |
| `losses.py` | Loss functions and metrics |
| `evaluate.py` | Single-GPU evaluation script |

## Configuration

Edit `config.py` to change hyperparameters:

```python
base_lr: float = 5e-5        # Learning rate (scaled by sqrt(world_size))
base_eta_min: float = 1e-6   # Minimum LR for cosine decay
num_epochs: int = 15          # Total epochs
warmup_epochs: int = 1        # Linear warmup epochs
base_batch_size: int = 64     # Per-GPU batch size (effective = 64 × 4 = 256)
gradient_accum_layers: int = 24  # Layers per backward pass (24 = all at once)
root_dir: str = "logs/distill_MO_v1"  # Checkpoints and metrics output
```

### LR Schedule

```
Epoch 1:     Linear warmup (1% → 100% of peak LR)
Epochs 2-15: Cosine annealing decay to eta_min
```

## Training on Compute Canada (Fir)

### Fresh start

```bash
sbatch submit_distill.sh
```

### Resume from checkpoint

```bash
RESUME=1 sbatch submit_distill.sh
```

Resumes from `<root_dir>/checkpoint_latest.pt`. Model weights, optimizer state, LR scheduler, and metrics history are all restored — training continues exactly where it left off.

### SLURM partitions (GPU)

| Partition | Max Time | Nodes | Use |
|-----------|----------|-------|-----|
| `gpubase_bynode_b1` | 3h | 80 | Quick tests, fastest scheduling |
| `gpubase_bynode_b2` | 12h | 80 | Medium runs |
| `gpubase_bynode_b3` | 24h | 62 | Full day training |
| `gpubase_bynode_b4` | 3 days | 40 | Multi-day runs |
| `gpubase_bynode_b5` | 7 days | 26 | Week-long runs |
| `gpubase_bygpu_b1-b5` | Same tiers | More nodes | Single-GPU jobs (eval) |

Use `bynode` for training (4 GPUs) and `bygpu` for evaluation (1 GPU).

### Monitoring a running job

```bash
# Check job status
squeue -u $USER

# Watch training output live
tail -f logs/distill_MO_mambaformer_<jobid>.out

# Cancel a job
scancel <jobid>
```

### Splitting long runs

At ~44 min/epoch, use this to plan wall time:

| Epochs | Training Time | + Extraction | Partition |
|--------|--------------|--------------|-----------|
| 3 | ~2h 12min | ~2h 18min | b1 (3h) |
| 5 | ~3h 40min | ~3h 46min | b2 (12h) |
| 11 | ~8h 04min | ~8h 10min | b2 (12h) |
| 15 | ~11h | ~11h 06min | b2 (12h) |

For long runs, submit on `b2` then resume if needed. Checkpoints are saved every epoch.

## Evaluation

Evaluate a checkpoint on the validation set (single GPU, ~1 min):

```bash
# Evaluate latest checkpoint
sbatch submit_eval.sh

# Evaluate a specific epoch
CKPT=logs/distill_MO_v1/checkpoint_ep4.pt sbatch submit_eval.sh
```

Output is printed as a per-layer table and saved to `eval_<checkpoint_name>.json` in the checkpoint directory.

## Output files

All saved under `root_dir` (default: `logs/distill_MO_v1/`):

```
logs/distill_MO_v1/
├── checkpoint_ep1.pt          # Per-epoch checkpoint
├── checkpoint_ep2.pt
├── ...
├── checkpoint_latest.pt       # Latest checkpoint (for resume)
├── training_metrics.json      # All epochs metrics (train + val)
├── eval_checkpoint_ep4.json   # Evaluation results
└── eval_checkpoint_latest.json
```

## Data

Training and validation tars are extracted to local NVMe (`$SLURM_TMPDIR`) at job start:

- **Train**: `/project/def-mpederso/dataset/imagenet/imagenet.tar` (145GB, ~5 min extraction)
- **Val**: `/project/def-mpederso/dataset/imagenet_val.tar` (6.3GB, ~30 sec extraction)

The eval script only extracts the val tar.
