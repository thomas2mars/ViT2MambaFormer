from dataclasses import dataclass


@dataclass
class DistillationConfig:
    # Training
    base_batch_size: int = 32
    base_lr: float = 5e-5
    base_eta_min: float = 1e-6
    num_epochs: int = 10
    warmup_epochs: int = 1

    # How many layers to accumulate before calling .backward()
    # 1 = Backward every layer (Lowest Memory, Slowest)
    # 4 = Backward every 4 layers (Higher Memory, Faster)
    # 12 = Backward once per image (Highest Memory, Fastest)
    gradient_accum_layers: int = 12

    # Logging
    heavy_metrics_logs_per_epoch: int = 10

    # Distillation
    num_layers: int = 12
    double_cls_token: bool = True
    temp: float = 1.0

    # Paths
    root_dir: str = "distillation_b_wt/MO/logs/distill_MO_base_v1"
    teacher_weights_path: str = "models/vit/ViT-B_16.npz"
    data_dir: str = "dataset/ImageNet_ILSVRC2012"

    # DataLoader
    num_workers: int = 8
    prefetch_factor: int = 3

    @property
    def full_val_interval(self):
        return max(1, self.num_epochs // 5)

    @property
    def save_model_every(self):
        return max(1, self.num_epochs // 5)