"""WK Base v2 — Higher LR + longer schedule + lower temperature.
Teacher: ViT-B/16 (Google .npz, 84.16% top-1) | Student: MambaFormer_Base_expand1_light_BiMamba2
"""
from dataclasses import dataclass


@dataclass
class DistillationConfig:
    # Training
    base_batch_size: int = 32
    base_lr: float = 1e-5
    base_eta_min: float = 1e-7
    num_epochs: int = 50
    warmup_epochs: int = 4
    distillation_temp: float = 5.0

    # Loss
    kl_weight: float = 1.0
    ce_weight: float = 0.0
    label_smoothing: float = 0.0

    # Logging
    save_model_every: int = 5

    # Distillation
    double_cls_token: bool = True

    # Paths
    root_dir: str = "distillation_b_wt/WK/logs/distill_WK_base_v2"
    teacher_weights_path: str = "models/vit/ViT-B_16.npz"
    step2_checkpoint_path: str = "distillation_b_wt/HA/checkpoint_latest.pt"
    data_dir: str = "dataset/ImageNet_ILSVRC2012"

    # DataLoader
    num_workers: int = 8
    val_num_workers: int = 4

    # Loss mode: 'soft' (KL) or 'hard' (DeiT hard-label distillation)
    loss_type: str = 'soft'
    distill_alpha: float = 0.5  # weight on teacher CE in hard distillation

    # Augmentation (disabled by default)
    use_randaugment: bool = False
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 9
    use_random_erasing: bool = False
    random_erasing_p: float = 0.25
    use_mixup_cutmix: bool = False
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0

    @property
    def full_val_interval(self):
        return max(1, self.num_epochs // 11)
