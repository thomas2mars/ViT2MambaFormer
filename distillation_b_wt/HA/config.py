from dataclasses import dataclass


@dataclass
class DistillationConfig:
    # Training
    base_batch_size: int = 32
    base_lr: float = 5e-5
    eta_min: float = 1e-6
    num_epochs: int = 15
    warmup_epochs: int = 1
    grad_clip_norm: float = 5.0

    # How many layers to accumulate before calling .backward()
    gradient_accum_layers: int = 12

    # Logging
    heavy_metrics_logs_per_epoch: int = 10

    # Distillation
    num_layers: int = 12
    double_cls_token: bool = True
    temp: float = 1.0
    normalization: str = "softmax"

    # Paths
    root_dir: str = "distillation_b_wt/HA/logs/distill_HA_base_v1"
    teacher_weights_path: str = "models/vit/ViT-B_16.npz"
    step1_checkpoint_path: str = "distillation_b_wt/MO/logs/distill_MO_base_v1/checkpoint_ep9.pt"
    data_dir: str = "dataset/ImageNet_ILSVRC2012"

    # DataLoader
    num_workers: int = 8
    prefetch_factor: int = 3

    @property
    def full_val_interval(self):
        return max(1, self.num_epochs // 10)

    @property
    def save_model_every(self):
        return max(1, self.num_epochs // 10)
