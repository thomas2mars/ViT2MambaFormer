import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torchvision import datasets, transforms
from torchvision.models import vit_b_16
from tqdm import tqdm
import time
import sys
import os
import argparse

# --- IMPORTS ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))   # distillation_b_wt_10pct/ for WK.*
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))  # project root for utils.*
from utils.vit_utils import convert_npz_to_torchvision
from utils.data import get_class_stratified_subset
from utils.training import (
    setup_ddp, cleanup_ddp, save_checkpoint, load_checkpoint,
    save_metrics,
)
from MambaFormer.MambaFormer import MambaFormer_Base_expand1_light_BiMamba2
from WK.losses import logit_distillation_loss, hard_distillation_loss, compute_agreement

# Clear registry
try:
    from torchvision.models._api import BUILTIN_MODELS
    BUILTIN_MODELS.clear()
except ImportError:
    pass


# --- DATA ---

def build_dataloaders(cfg):
    """Build train/val dataloaders with Google ViT [-1, 1] normalization at 384x384.
    Training set is subsetted to cfg.subset_fraction per class."""
    train_ops = [
        transforms.RandomResizedCrop(384),
        transforms.RandomHorizontalFlip(),
    ]
    if cfg.use_randaugment:
        train_ops.append(transforms.RandAugment(num_ops=cfg.randaugment_num_ops,
                                                magnitude=cfg.randaugment_magnitude))
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    if cfg.use_random_erasing:
        train_ops.append(transforms.RandomErasing(p=cfg.random_erasing_p))
    train_transform = transforms.Compose(train_ops)
    val_transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(cfg.data_dir, "train"), transform=train_transform
    )
    train_dataset = get_class_stratified_subset(
        train_dataset, fraction=cfg.subset_fraction, seed=cfg.subset_seed
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(cfg.data_dir, "val"), transform=val_transform
    )

    distributed = dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.base_batch_size, sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=False, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.base_batch_size, sampler=val_sampler,
        num_workers=cfg.val_num_workers, pin_memory=False, persistent_workers=False
    )

    return train_loader, val_loader, train_sampler


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
    for p in teacher.parameters():
        p.requires_grad = False
    teacher = torch.compile(teacher, mode="reduce-overhead")
    return teacher


def load_student(cfg, device, local_rank):
    student = MambaFormer_Base_expand1_light_BiMamba2(double_cls_token=cfg.double_cls_token, image_size=384)

    # Load step 2 checkpoint
    if os.path.exists(cfg.step2_checkpoint_path):
        checkpoint = torch.load(cfg.step2_checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        student.load_state_dict(state_dict)
        print(f"Loaded step 2 checkpoint from {cfg.step2_checkpoint_path}")
    else:
        raise FileNotFoundError(f"Step 2 checkpoint not found at {cfg.step2_checkpoint_path}")

    student.to(device)
    student.train()

    # Freeze everything first
    for p in student.parameters():
        p.requires_grad = False

    # Unfreeze mixers (all Mamba components)
    for layer in student.encoder.layers:
        for param in layer.mixer.parameters():
            param.requires_grad = True

    # Unfreeze classification head
    for param in student.heads.parameters():
        param.requires_grad = True

    # Unfreeze CLS fusion and CLS tokens
    for param in student.cls_fusion.parameters():
        param.requires_grad = True
    student.forward_cls_token.requires_grad = True
    student.backward_cls_token.requires_grad = True

    if dist.is_initialized():
        student = DDP(student, device_ids=[local_rank], find_unused_parameters=False)

    return student


# --- AUGMENTATION ---

def apply_mixup_cutmix(images, labels, mixup_alpha=0.8, cutmix_alpha=1.0):
    """Apply Mixup or CutMix (50/50) to a batch.

    Returns:
        mixed_images, shuffled_labels, lam  — lam is the mixing coefficient
        for label blending: lam * CE(y_a) + (1-lam) * CE(y_b).
    """
    index = torch.randperm(images.size(0), device=images.device)
    labels_b = labels[index]

    if torch.rand(1).item() < 0.5:
        # Mixup
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        mixed = lam * images + (1 - lam) * images[index]
    else:
        # CutMix
        lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
        _, _, H, W = images.shape
        cut_rat = (1 - lam) ** 0.5
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()
        x1, y1 = max(0, cx - cut_w // 2), max(0, cy - cut_h // 2)
        x2, y2 = min(W, cx + cut_w // 2), min(H, cy + cut_h // 2)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        # Adjust lam to actual area ratio
        lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)

    return mixed, labels_b, lam


# --- TRAINING / VALIDATION ---

def train_one_epoch(student, teacher, optimizer, dataloader, cfg, device, is_master, dtype):
    student.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    steps = 0

    interactive = not os.environ.get('SLURM_JOB_ID')
    progress_bar = tqdm(dataloader, desc='Training', unit='batch', disable=not (is_master and interactive))

    for batch, labels in progress_bar:
        batch = batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        labels_b, lam = None, 1.0
        if cfg.use_mixup_cutmix:
            batch, labels_b, lam = apply_mixup_cutmix(batch, labels, cfg.mixup_alpha, cfg.cutmix_alpha)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=True, dtype=dtype):
            with torch.no_grad():
                teacher_logits = teacher(batch)
            student_logits = student(batch)

            if cfg.loss_type == 'hard':
                loss = hard_distillation_loss(
                    student_logits, teacher_logits, labels, labels_b, lam,
                    alpha=cfg.distill_alpha, label_smoothing=cfg.label_smoothing
                )
            else:
                loss = logit_distillation_loss(
                    student_logits, teacher_logits, labels,
                    temperature=cfg.distillation_temp,
                    kl_weight=cfg.kl_weight, ce_weight=cfg.ce_weight,
                    label_smoothing=cfg.label_smoothing
                )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct, total = compute_agreement(student_logits, teacher_logits)
        train_correct += correct
        train_total += total
        steps += 1

        if is_master:
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

    # DDP aggregation
    if dist.is_initialized():
        metrics_tensor = torch.tensor([train_correct, train_total], device=device, dtype=torch.long)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        train_correct = metrics_tensor[0].item()
        train_total = metrics_tensor[1].item()

        loss_tensor = torch.tensor(running_loss / steps, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
    else:
        avg_loss = running_loss / steps

    accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0
    return avg_loss, accuracy


def validate(student, teacher, val_dataloader, cfg, device, is_master, dtype):
    student.eval()
    val_loss_accum = 0.0
    val_correct = 0
    val_total = 0
    val_steps = 0

    with torch.no_grad():
        interactive = not os.environ.get('SLURM_JOB_ID')
        for batch, labels in tqdm(val_dataloader, desc="Validating", disable=not (is_master and interactive)):
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=True, dtype=dtype):
                teacher_logits = teacher(batch)
                student_logits = student(batch)

                if cfg.loss_type == 'hard':
                    loss = hard_distillation_loss(
                        student_logits, teacher_logits, labels,
                        alpha=cfg.distill_alpha, label_smoothing=cfg.label_smoothing
                    )
                else:
                    loss = logit_distillation_loss(
                        student_logits, teacher_logits, labels,
                        temperature=cfg.distillation_temp,
                        kl_weight=cfg.kl_weight, ce_weight=cfg.ce_weight,
                        label_smoothing=cfg.label_smoothing
                    )

            val_loss_accum += loss.item()
            correct, total = compute_agreement(student_logits, teacher_logits)
            val_correct += correct
            val_total += total
            val_steps += 1

    # DDP aggregation
    if dist.is_initialized():
        metrics_tensor = torch.tensor([val_correct, val_total], device=device, dtype=torch.long)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        val_correct = metrics_tensor[0].item()
        val_total = metrics_tensor[1].item()

        loss_tensor = torch.tensor(val_loss_accum / val_steps, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
    else:
        avg_loss = val_loss_accum / val_steps

    accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0
    return avg_loss, accuracy


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config',
                        help='Config module name under WK/ (e.g., config, config_aug)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to ImageNet dataset (use $SLURM_TMPDIR for local NVMe)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from latest checkpoint')
    args = parser.parse_args()

    import importlib
    config_module = importlib.import_module(f'WK.{args.config}')
    cfg = config_module.DistillationConfig()
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir

    local_rank, global_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_master = (global_rank == 0)
    dtype = torch.bfloat16

    # Linear LR scaling
    learning_rate = cfg.base_lr * world_size
    min_eta = cfg.base_eta_min * world_size
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    if is_master:
        os.makedirs(cfg.root_dir, exist_ok=True)
        print(f"=== Step 3: End-to-End Logit Distillation (10% subset, seed={cfg.subset_seed}) ===")
        print(f"World size: {world_size} | Batch per GPU: {cfg.base_batch_size} | Effective batch: {cfg.base_batch_size * world_size}")
        print(f"LR: {learning_rate:.2e} (base {cfg.base_lr:.2e} x {world_size}) | Warmup: {cfg.warmup_epochs} epochs | Temperature: {cfg.distillation_temp}")

    # --- Models ---
    teacher = load_teacher(cfg, device)
    student = load_student(cfg, device, local_rank)
    student_ref = student.module if dist.is_initialized() else student

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    if is_master:
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)} / {sum(p.numel() for p in student.parameters())}")

    optimizer = optim.AdamW(trainable_params, lr=learning_rate)

    # Warmup + Cosine Annealing
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs - cfg.warmup_epochs, eta_min=min_eta
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

    # --- Data ---
    train_loader, val_loader, train_sampler = build_dataloaders(cfg)

    if is_master:
        print(f"Training on {len(train_loader.dataset)} samples ({cfg.subset_fraction*100:.0f}% subset)")

    try:
        for epoch in range(start_epoch, cfg.num_epochs):
            if dist.is_initialized():
                train_sampler.set_epoch(epoch)

            # --- Train ---
            train_start = time.time()
            avg_train_loss, train_accuracy = train_one_epoch(
                student, teacher, optimizer, train_loader, cfg, device, is_master, dtype
            )
            train_time = time.time() - train_start

            scheduler.step()

            if is_master:
                print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Agreement: {train_accuracy:.2f}% | Train Time: {train_time:.1f}s")

            # --- Validate ---
            val_loss = None
            val_accuracy = None
            val_time = 0.0
            if (epoch + 1) % cfg.full_val_interval == 0 or (epoch == cfg.num_epochs - 1):
                val_start = time.time()
                val_loss, val_accuracy = validate(
                    student, teacher, val_loader, cfg, device, is_master, dtype
                )
                val_time = time.time() - val_start

                if is_master:
                    print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val Agreement: {val_accuracy:.2f}% | Val Time: {val_time:.1f}s")

            # --- Log & Save ---
            if is_master:
                epoch_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'train_accuracy': train_accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                all_epoch_metrics.append(epoch_metrics)

                metrics_file = save_metrics(all_epoch_metrics, cfg.root_dir)
                print(f"Epoch {epoch+1} metrics saved to {metrics_file}")

                # Save checkpoint every epoch for crash resilience
                save_checkpoint(student_ref, optimizer, scheduler, epoch + 1, all_epoch_metrics, cfg.root_dir)

                # Save standalone model weights at intervals
                if (epoch + 1) % cfg.save_model_every == 0 or (epoch == cfg.num_epochs - 1):
                    torch.save(student_ref.state_dict(), os.path.join(cfg.root_dir, f"model_ep{epoch+1}.pth"))
                    print(f"Epoch {epoch+1} model saved")

    finally:
        pass

    cleanup_ddp()


if __name__ == "__main__":
    main()
