import math
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import vit_b_16
from tqdm import tqdm
import time
import sys
import os
import argparse

# --- IMPORTS ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))   # distillation_b_wt_10pct/ for HA.*
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))  # project root for utils.*
from utils.vit_utils import ViTStatesExtractor, convert_npz_to_torchvision
from utils.data import build_subset_dataloaders
from utils.training import (
    setup_ddp, cleanup_ddp, save_checkpoint, load_checkpoint,
    aggregate_layer_metrics, save_metrics, init_layer_trackers,
)
from MambaFormer.MambaFormer import MambaFormer_Base_expand1_light_BiMamba2
from MambaFormer.utils import (
    compute_ssd_attention_map,
    hard_reset_z_gates,
    zero_init_output_projections,
    patch_student_for_extraction,
)
from HA.config import DistillationConfig
from HA.losses import (
    combined_output_attention_loss, linear_cka,
)

# Clear registry
try:
    from torchvision.models._api import BUILTIN_MODELS
    BUILTIN_MODELS.clear()
except ImportError:
    pass

# Metric keys for step 2 (output alignment + attention)
HA_TRACKER_KEYS = ['losses', 'l2_losses', 'fro_losses', 'cka_metrics']
HA_METRIC_NAMES = {
    'losses': 'loss',
    'l2_losses': 'l2',
    'fro_losses': 'fro',
    'cka_metrics': 'cka',
}


# --- MODEL SETUP ---

def load_teacher(cfg, device):
    teacher = vit_b_16(weights=None, image_size=384)
    if os.path.exists(cfg.teacher_weights_path):
        if cfg.teacher_weights_path.endswith('.npz'):
            state_dict = convert_npz_to_torchvision(
                cfg.teacher_weights_path, num_layers=12, embed_dim=768, num_heads=12
            )
        else:
            state_dict = torch.load(cfg.teacher_weights_path, map_location='cpu', weights_only=True)
        teacher.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Teacher weights not found at {cfg.teacher_weights_path}")

    teacher.to(device)
    teacher.eval()
    teacher = torch.compile(teacher, mode="reduce-overhead")

    extractor = ViTStatesExtractor(
        teacher, layer_indices=None, extract_attention=True,
        double_cls_token=cfg.double_cls_token
    )
    return teacher, extractor


def load_student(cfg, device, local_rank):
    student = MambaFormer_Base_expand1_light_BiMamba2(double_cls_token=cfg.double_cls_token, image_size=384)
    separate_directions = student.separate_directions

    # Load step 1 checkpoint
    if os.path.exists(cfg.step1_checkpoint_path):
        checkpoint = torch.load(cfg.step1_checkpoint_path, map_location='cpu', weights_only=False)
        student.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded step 1 checkpoint from {cfg.step1_checkpoint_path}")
    else:
        raise FileNotFoundError(f"Step 1 checkpoint not found at {cfg.step1_checkpoint_path}")

    # Reset z-gates and output projections for step 2 training
    hard_reset_z_gates(student, layer_indices=list(range(cfg.num_layers)), separate_directions=separate_directions)
    zero_init_output_projections(student, layer_indices=list(range(cfg.num_layers)))

    student.to(device)
    student.train()

    # Freeze everything, then unfreeze trainable components for step 2:
    # - in_proj (produces B, C, w for attention maps)
    # - out_proj on each mamba direction + BiMamba out_proj (output alignment)
    # - down/up projections if bottleneck is used
    for p in student.parameters():
        p.requires_grad = False
    for layer in student.encoder.layers:
        if student.hidden_dim != student.mamba_hidden_dim:
            for param in layer.down_proj.parameters():
                param.requires_grad = True
            for param in layer.up_proj.parameters():
                param.requires_grad = True
        mixer = layer.mixer
        if separate_directions:
            for param in mixer.fwd_mamba.in_proj.parameters():
                param.requires_grad = True
            for param in mixer.bwd_mamba.in_proj.parameters():
                param.requires_grad = True
            for param in mixer.fwd_mamba.out_proj.parameters():
                param.requires_grad = True
            for param in mixer.bwd_mamba.out_proj.parameters():
                param.requires_grad = True
        else:
            for param in mixer.mamba.in_proj.parameters():
                param.requires_grad = True
            for param in mixer.mamba.out_proj.parameters():
                param.requires_grad = True
        for param in mixer.out_proj.parameters():
            param.requires_grad = True

    if dist.is_initialized():
        student = DDP(student, device_ids=[local_rank], find_unused_parameters=False)

    return student, separate_directions


def student_layer_forward(student_ref, layer_idx, layer_input):
    """Forward through a single student layer: LN -> [down_proj ->] mixer [-> up_proj]."""
    layer = student_ref.encoder.layers[layer_idx]
    ln_output = layer.ln_1(layer_input)
    if student_ref.hidden_dim != student_ref.mamba_hidden_dim:
        ln_output = layer.down_proj(ln_output)
    mixer_output = layer.mixer(ln_output)
    if student_ref.hidden_dim != student_ref.mamba_hidden_dim:
        mixer_output = layer.up_proj(mixer_output)
    return mixer_output


# --- TRAINING / VALIDATION ---

def train_one_epoch(student, student_ref, teacher_extractor, optimizer, dataloader,
                    get_layer_states, clear_states, cfg, device, is_master, dtype):
    student.train()

    num_layers = cfg.num_layers
    metrics_interval = max(1, len(dataloader) // cfg.heavy_metrics_logs_per_epoch)
    trackers = init_layer_trackers(num_layers, keys=HA_TRACKER_KEYS)
    epoch_loss = 0.0

    interactive = not os.environ.get('SLURM_JOB_ID')
    progress_bar = tqdm(dataloader, desc='Training', unit='batch', disable=not (is_master and interactive))

    for batch_idx, (batch, _) in enumerate(progress_bar):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        clear_states()

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=dtype):
            _, teacher_states = teacher_extractor.get_vit_states(batch)

        batch_loss_tensor = torch.zeros(1, device=device, dtype=torch.float32)
        chunk_loss = 0.0

        compute_metrics = is_master and (batch_idx % metrics_interval == 0)

        layer_loss_tensors = []
        layer_l2_tensors = []
        layer_fro_tensors = []
        layer_output_pairs = []

        # --- CHUNKED LAYER ITERATION ---
        with torch.amp.autocast('cuda', enabled=True, dtype=dtype):
            for layer_idx in range(num_layers):

                # 1. Input/Target Selection
                if layer_idx == 0:
                    inp = teacher_states['first_layer_input']
                else:
                    inp = teacher_states['layers_output'][f'layer_{layer_idx - 1}']

                layer_input = inp.detach()
                teacher_attn = teacher_states['attention_maps'][f'layer_{layer_idx}']
                teacher_mixer_output = teacher_states['layers_mixer_output'][f'layer_{layer_idx}']

                # 2. Forward (LN -> mixer path, patched layer captures states)
                student_mixer_output = student_layer_forward(student_ref, layer_idx, layer_input)
                layer_states = get_layer_states(layer_idx)

                # 3. Compute student attention map
                _, _, student_attn = compute_ssd_attention_map(
                    layer_states, weighted=True, normalization=cfg.normalization,
                    per_head=True, temp=cfg.temp
                )

                # 4. Combined loss (0.7 * L2 output + 0.3 * Frobenius attention)
                loss, loss_l2, loss_fro = combined_output_attention_loss(
                    student_mixer_output, teacher_mixer_output, student_attn, teacher_attn
                )

                if compute_metrics:
                    layer_loss_tensors.append(loss.detach())
                    layer_l2_tensors.append(loss_l2.detach())
                    layer_fro_tensors.append(loss_fro.detach())
                    layer_output_pairs.append((student_mixer_output.detach(), teacher_mixer_output.detach()))

                # 5. Accumulate Loss into Chunk
                chunk_loss = chunk_loss + loss

                # 6. Check if we should Backward
                is_last_layer = (layer_idx == num_layers - 1)
                if (layer_idx + 1) % cfg.gradient_accum_layers == 0 or is_last_layer:
                    chunk_loss.backward()
                    batch_loss_tensor += chunk_loss.detach()
                    chunk_loss = 0.0

        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip_norm)
        optimizer.step()

        total_batch_loss = batch_loss_tensor.item()
        epoch_loss += total_batch_loss

        if compute_metrics:
            for layer_idx in range(num_layers):
                layer_key = f'layer_{layer_idx}'
                trackers['losses'][layer_key].append(layer_loss_tensors[layer_idx].item())
                trackers['l2_losses'][layer_key].append(layer_l2_tensors[layer_idx].item())
                trackers['fro_losses'][layer_key].append(layer_fro_tensors[layer_idx].item())
                s_out, t_out = layer_output_pairs[layer_idx]
                trackers['cka_metrics'][layer_key].append(linear_cka(s_out.float(), t_out.float()).item())

        if is_master:
            progress_bar.set_postfix({'Avg Layer Loss': f'{total_batch_loss / num_layers:.4f}'})

    return epoch_loss, trackers


def validate(student, student_ref, teacher_extractor, val_dataloader,
             get_layer_states, clear_states, cfg, device, is_master, dtype):
    student.eval()

    num_layers = cfg.num_layers
    trackers = init_layer_trackers(num_layers, keys=HA_TRACKER_KEYS)
    val_loss_tensor = torch.zeros(1, device=device, dtype=torch.float32)
    val_steps = 0

    with torch.no_grad():
        interactive = not os.environ.get('SLURM_JOB_ID')
        for batch, _ in tqdm(val_dataloader, desc="Validating", disable=not (is_master and interactive)):
            batch = batch.to(device, non_blocking=True)
            clear_states()

            with torch.amp.autocast('cuda', enabled=True, dtype=dtype):
                _, t_states = teacher_extractor.get_vit_states(batch)

                for layer_idx in range(num_layers):
                    if layer_idx == 0:
                        inp = t_states['first_layer_input']
                    else:
                        inp = t_states['layers_output'][f'layer_{layer_idx - 1}']

                    t_attn = t_states['attention_maps'][f'layer_{layer_idx}']
                    t_mixer_output = t_states['layers_mixer_output'][f'layer_{layer_idx}']

                    s_mixer_output = student_layer_forward(student_ref, layer_idx, inp)
                    s_states = get_layer_states(layer_idx)
                    _, _, s_attn = compute_ssd_attention_map(
                        s_states, weighted=True, normalization=cfg.normalization,
                        per_head=True, temp=cfg.temp
                    )

                    v_loss, v_l2, v_fro = combined_output_attention_loss(
                        s_mixer_output, t_mixer_output, s_attn, t_attn
                    )
                    val_loss_tensor += v_loss

                    if is_master:
                        layer_key = f'layer_{layer_idx}'
                        trackers['losses'][layer_key].append(v_loss.item())
                        trackers['l2_losses'][layer_key].append(v_l2.item())
                        trackers['fro_losses'][layer_key].append(v_fro.item())
                        cka = linear_cka(s_mixer_output.float(), t_mixer_output.float())
                        trackers['cka_metrics'][layer_key].append(cka.item())

            val_steps += 1

    avg_val_loss = (val_loss_tensor / (val_steps * num_layers)).item()
    if dist.is_initialized():
        tens = torch.tensor(avg_val_loss, device=device)
        dist.all_reduce(tens, op=dist.ReduceOp.AVG)
        avg_val_loss = tens.item()

    return avg_val_loss, trackers


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to ImageNet dataset (use $SLURM_TMPDIR for local NVMe)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from latest checkpoint')
    args = parser.parse_args()

    cfg = DistillationConfig()
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir

    local_rank, global_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_master = (global_rank == 0)
    dtype = torch.bfloat16

    # Step 2 uses linear LR scaling (not sqrt like step 1)
    learning_rate = cfg.base_lr * world_size
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if is_master:
        os.makedirs(cfg.root_dir, exist_ok=True)
        print(f"=== 10% ImageNet Subset Experiment (seed={cfg.subset_seed}) ===")
        print(f"World size: {world_size} | Batch per GPU: {cfg.base_batch_size} | Effective batch: {cfg.base_batch_size * world_size}")
        print(f"LR: {learning_rate:.2e} (base {cfg.base_lr:.2e} x {world_size}) | Grad clip: {cfg.grad_clip_norm}")
        print(f"Distilling {cfg.num_layers} layers. Chunk Size: {cfg.gradient_accum_layers}")

    # --- Models ---
    teacher, teacher_extractor = load_teacher(cfg, device)
    student, separate_directions = load_student(cfg, device, local_rank)
    student_ref = student.module if dist.is_initialized() else student

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    if is_master:
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)} / {sum(p.numel() for p in student.parameters())}")

    weight_decay = getattr(cfg, 'weight_decay', 0.01)
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    # Warmup + Cosine Annealing
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs - cfg.warmup_epochs, eta_min=cfg.eta_min
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[cfg.warmup_epochs]
    )

    # --- Resume from checkpoint if requested ---
    start_epoch = 0
    all_epoch_metrics = []
    if args.resume:
        start_epoch, all_epoch_metrics = load_checkpoint(
            cfg.root_dir, student_ref, optimizer, scheduler, device
        )
        if is_master:
            if start_epoch > 0:
                print(f"Resumed from checkpoint at epoch {start_epoch}")
            else:
                print("No checkpoint found, starting from scratch")

    # --- Data (384x384, Google ViT [-1,1] normalization, 10% subset) ---
    train_loader, val_loader, train_sampler = build_subset_dataloaders(
        cfg.data_dir, cfg.base_batch_size,
        subset_fraction=cfg.subset_fraction, subset_seed=cfg.subset_seed,
        num_workers=cfg.num_workers, prefetch_factor=cfg.prefetch_factor,
        image_size=384, normalization='google_vit'
    )

    if is_master:
        print(f"Training on {len(train_loader.dataset)} samples ({cfg.subset_fraction*100:.0f}% subset)")

    # --- Save initial student weights (before step 2 training) ---
    if is_master and start_epoch == 0:
        torch.save(student_ref.state_dict(), os.path.join(cfg.root_dir, "original_student.pth"))

    # --- Patch student for state extraction ---
    unpatch_student, get_layer_states, clear_states = patch_student_for_extraction(
        student_ref, cfg.num_layers, separate_directions=separate_directions
    )

    try:
        for epoch in range(start_epoch, cfg.num_epochs):
            if dist.is_initialized():
                train_sampler.set_epoch(epoch)

            # --- Train ---
            train_start = time.time()
            epoch_loss, train_trackers = train_one_epoch(
                student, student_ref, teacher_extractor, optimizer, train_loader,
                get_layer_states, clear_states, cfg, device, is_master, dtype
            )
            train_time = time.time() - train_start

            scheduler.step()

            # --- Validate ---
            val_metrics = {}
            val_time = 0.0
            if (epoch + 1) % cfg.full_val_interval == 0 or (epoch == cfg.num_epochs - 1):
                val_start = time.time()
                avg_val_loss, val_trackers = validate(
                    student, student_ref, teacher_extractor, val_loader,
                    get_layer_states, clear_states, cfg, device, is_master, dtype
                )
                val_time = time.time() - val_start

                if is_master:
                    val_metrics = aggregate_layer_metrics(val_trackers, cfg.num_layers, metric_names=HA_METRIC_NAMES)

            # --- Log & Save ---
            if is_master:
                train_metrics = aggregate_layer_metrics(train_trackers, cfg.num_layers, metric_names=HA_METRIC_NAMES)

                if 'average' in train_metrics:
                    avg = train_metrics['average']
                    print(f"Epoch {epoch+1} | Train Loss: {avg['loss']:.4f} | L2: {avg['l2']:.4f} | Fro: {avg['fro']:.4f} | CKA: {avg['cka']:.4f} | Train Time: {train_time:.1f}s")

                if 'average' in val_metrics:
                    avg = val_metrics['average']
                    print(f"Epoch {epoch+1} | Val Loss: {avg['loss']:.4f} | L2: {avg['l2']:.4f} | Fro: {avg['fro']:.4f} | CKA: {avg['cka']:.4f} | Val Time: {val_time:.1f}s")

                epoch_metrics = {
                    'epoch': epoch + 1,
                    'train': train_metrics,
                    'validation': val_metrics if val_metrics else None,
                    'learning_rate': scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else learning_rate
                }
                all_epoch_metrics.append(epoch_metrics)

                metrics_file = save_metrics(all_epoch_metrics, cfg.root_dir)
                print(f"Epoch {epoch+1} metrics saved to {metrics_file}")

                # Save checkpoint every epoch for crash resilience
                save_checkpoint(student_ref, optimizer, scheduler, epoch + 1, all_epoch_metrics, cfg.root_dir)
                print(f"Epoch {epoch+1} checkpoint saved")

    finally:
        unpatch_student()

    cleanup_ddp()


if __name__ == "__main__":
    main()
