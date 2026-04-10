import os
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Google ViT preprocessing: pixels mapped to [-1, 1]
GOOGLE_VIT_MEAN = [0.5, 0.5, 0.5]
GOOGLE_VIT_STD = [0.5, 0.5, 0.5]


def get_train_transform(image_size=224, normalization='imagenet'):
    mean, std = _get_norm_stats(normalization)
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def get_val_transform(image_size=384, normalization='imagenet'):
    mean, std = _get_norm_stats(normalization)
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def _get_norm_stats(normalization):
    if normalization == 'imagenet':
        return IMAGENET_MEAN, IMAGENET_STD
    elif normalization == 'google_vit':
        return GOOGLE_VIT_MEAN, GOOGLE_VIT_STD
    else:
        raise ValueError(f"Unknown normalization: {normalization}. Use 'imagenet' or 'google_vit'.")


def get_class_stratified_subset(dataset, fraction=0.1, seed=42):
    """Return a Subset with `fraction` of samples per class, deterministically."""
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    classes = np.unique(targets)

    selected_indices = []
    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        n_select = max(1, int(len(cls_indices) * fraction))
        selected = rng.choice(cls_indices, size=n_select, replace=False)
        selected_indices.extend(selected.tolist())

    selected_indices.sort()
    return Subset(dataset, selected_indices)


def build_subset_dataloaders(data_dir, batch_size, subset_fraction=0.1, subset_seed=42,
                             num_workers=12, prefetch_factor=3,
                             image_size=224, normalization='imagenet'):
    """Build train/val dataloaders with a class-stratified subset of the training set.

    Only the training set is subsetted; validation uses all samples.
    """
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=get_train_transform(image_size, normalization)
    )
    train_dataset = get_class_stratified_subset(train_dataset, fraction=subset_fraction, seed=subset_seed)

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=get_val_transform(image_size, normalization)
    )

    distributed = dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=prefetch_factor
    )

    return train_loader, val_loader, train_sampler


def build_dataloaders(data_dir, batch_size, num_workers=12, prefetch_factor=3,
                      image_size=224, normalization='imagenet'):
    """Build train and validation dataloaders with optional DDP samplers.

    Args:
        normalization: 'imagenet' for standard ImageNet stats, 'google_vit' for [-1, 1].

    Returns:
        (train_loader, val_loader, train_sampler)
    """
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=get_train_transform(image_size, normalization)
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=get_val_transform(image_size, normalization)
    )

    distributed = dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=prefetch_factor
    )

    return train_loader, val_loader, train_sampler


def build_val_dataloader(data_dir, batch_size, num_workers=12, prefetch_factor=3,
                         image_size=384, normalization='imagenet'):
    """Build validation-only dataloader (no train dataset needed)."""
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=get_val_transform(image_size, normalization)
    )

    return DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
