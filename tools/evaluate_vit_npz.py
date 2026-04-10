"""
Evaluate Google's ViT-L/16 .npz checkpoint (ImageNet21K+1K fine-tuned) on ImageNet1K.

Converts the JAX/numpy weights to torchvision's ViT-L/16 format, then evaluates
Top-1/Top-5 accuracy at 384x384 resolution.

Expected result: ~85.15% Top-1 accuracy (from the ViT paper, Table 2).

Usage:
    python evaluate_vit_npz.py --npz models/vit/ViT-L_16.npz --data_dir path/to/ImageNet
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import vit_l_16
from torch.utils.data import DataLoader
from tqdm import tqdm


def convert_npz_to_torchvision(npz_path):
    """Convert Google ViT .npz weights to torchvision state_dict."""
    w = np.load(npz_path)
    state_dict = {}

    # Patch embedding: (H, W, C_in, C_out) -> (C_out, C_in, H, W)
    state_dict['conv_proj.weight'] = torch.from_numpy(
        w['embedding/kernel'].transpose(3, 2, 0, 1).copy()
    )
    state_dict['conv_proj.bias'] = torch.from_numpy(w['embedding/bias'].copy())

    # CLS token
    state_dict['class_token'] = torch.from_numpy(w['cls'].copy())

    # Positional embedding
    state_dict['encoder.pos_embedding'] = torch.from_numpy(
        w['Transformer/posembed_input/pos_embedding'].copy()
    )

    # Final encoder LayerNorm
    state_dict['encoder.ln.weight'] = torch.from_numpy(
        w['Transformer/encoder_norm/scale'].copy()
    )
    state_dict['encoder.ln.bias'] = torch.from_numpy(
        w['Transformer/encoder_norm/bias'].copy()
    )

    # Classification head
    state_dict['heads.head.weight'] = torch.from_numpy(
        w['head/kernel'].transpose().copy()
    )
    state_dict['heads.head.bias'] = torch.from_numpy(w['head/bias'].copy())

    # Encoder blocks (24 layers for ViT-L)
    for i in range(24):
        prefix_jax = f'Transformer/encoderblock_{i}'
        prefix_pt = f'encoder.layers.encoder_layer_{i}'

        # LayerNorm 1 (pre-attention)
        state_dict[f'{prefix_pt}.ln_1.weight'] = torch.from_numpy(
            w[f'{prefix_jax}/LayerNorm_0/scale'].copy()
        )
        state_dict[f'{prefix_pt}.ln_1.bias'] = torch.from_numpy(
            w[f'{prefix_jax}/LayerNorm_0/bias'].copy()
        )

        # LayerNorm 2 (pre-MLP)
        state_dict[f'{prefix_pt}.ln_2.weight'] = torch.from_numpy(
            w[f'{prefix_jax}/LayerNorm_2/scale'].copy()
        )
        state_dict[f'{prefix_pt}.ln_2.bias'] = torch.from_numpy(
            w[f'{prefix_jax}/LayerNorm_2/bias'].copy()
        )

        # Self-attention: Q, K, V kernels (1024, 16, 64) -> reshape to (1024, 1024) -> transpose
        attn_prefix = f'{prefix_jax}/MultiHeadDotProductAttention_1'
        q_w = w[f'{attn_prefix}/query/kernel'].reshape(1024, 1024).transpose()
        k_w = w[f'{attn_prefix}/key/kernel'].reshape(1024, 1024).transpose()
        v_w = w[f'{attn_prefix}/value/kernel'].reshape(1024, 1024).transpose()
        # in_proj_weight = [Q; K; V] shape (3072, 1024)
        state_dict[f'{prefix_pt}.self_attention.in_proj_weight'] = torch.from_numpy(
            np.concatenate([q_w, k_w, v_w], axis=0).copy()
        )

        # Self-attention: Q, K, V biases (16, 64) -> reshape to (1024,)
        q_b = w[f'{attn_prefix}/query/bias'].reshape(1024)
        k_b = w[f'{attn_prefix}/key/bias'].reshape(1024)
        v_b = w[f'{attn_prefix}/value/bias'].reshape(1024)
        state_dict[f'{prefix_pt}.self_attention.in_proj_bias'] = torch.from_numpy(
            np.concatenate([q_b, k_b, v_b], axis=0).copy()
        )

        # Output projection: (16, 64, 1024) -> reshape to (1024, 1024) -> transpose
        state_dict[f'{prefix_pt}.self_attention.out_proj.weight'] = torch.from_numpy(
            w[f'{attn_prefix}/out/kernel'].reshape(1024, 1024).transpose().copy()
        )
        state_dict[f'{prefix_pt}.self_attention.out_proj.bias'] = torch.from_numpy(
            w[f'{attn_prefix}/out/bias'].copy()
        )

        # MLP
        mlp_prefix = f'{prefix_jax}/MlpBlock_3'
        state_dict[f'{prefix_pt}.mlp.linear_1.weight'] = torch.from_numpy(
            w[f'{mlp_prefix}/Dense_0/kernel'].transpose().copy()
        )
        state_dict[f'{prefix_pt}.mlp.linear_1.bias'] = torch.from_numpy(
            w[f'{mlp_prefix}/Dense_0/bias'].copy()
        )
        state_dict[f'{prefix_pt}.mlp.linear_2.weight'] = torch.from_numpy(
            w[f'{mlp_prefix}/Dense_1/kernel'].transpose().copy()
        )
        state_dict[f'{prefix_pt}.mlp.linear_2.bias'] = torch.from_numpy(
            w[f'{mlp_prefix}/Dense_1/bias'].copy()
        )

    return state_dict


def build_google_vit_val_loader(data_dir, batch_size, image_size=384, num_workers=8):
    """
    Google ViT preprocessing: resize smaller side to image_size, center-crop,
    then normalize to [-1, 1] (mean=0.5, std=0.5) — NOT ImageNet mean/std.
    This matches the original vit_jax evaluation pipeline.
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),          # resize shorter edge to 384
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
    ])
    dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=3,
    )


@torch.no_grad()
def evaluate(model, val_loader, device, dtype):
    correct_1 = 0
    correct_5 = 0
    total = 0

    interactive = not os.environ.get('SLURM_JOB_ID')
    for images, targets in tqdm(val_loader, desc="Evaluating", disable=not interactive):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=True, dtype=dtype):
            logits = model(images)

        pred_1 = logits.argmax(dim=1)
        correct_1 += (pred_1 == targets).sum().item()

        _, pred_5 = logits.topk(5, dim=1)
        correct_5 += (pred_5 == targets.unsqueeze(1)).any(dim=1).sum().item()

        total += targets.size(0)

    top1 = 100.0 * correct_1 / total
    top5 = 100.0 * correct_5 / total
    return {'top1': top1, 'top5': top5, 'total_samples': total}


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Google ViT-L/16 .npz checkpoint on ImageNet1K'
    )
    parser.add_argument('--npz', type=str, default='models/vit/ViT-L_16.npz',
                        help='Path to Google .npz checkpoint')
    parser.add_argument('--data_dir', type=str, default='dataset/ImageNet_ILSVRC2012')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--output', type=str, default='logs/eval_vit_npz_google_norm.json')
    args = parser.parse_args()

    # Step 1: Convert .npz to torchvision state_dict (CPU only, no GPU needed)
    print(f"Converting {args.npz} to torchvision format...")
    state_dict = convert_npz_to_torchvision(args.npz)
    print(f"Converted {len(state_dict)} parameters.")

    # Step 2: Build model and load weights
    device = torch.device('cuda:0')
    dtype = torch.bfloat16
    torch.set_float32_matmul_precision("high")

    model = vit_l_16(weights=None, image_size=args.image_size)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"WARNING: Missing keys: {missing}")
    if unexpected:
        print(f"WARNING: Unexpected keys: {unexpected}")
    print("All weights loaded successfully.")

    model.to(device)
    model.eval()

    # Step 3: Evaluate with Google ViT preprocessing ([-1, 1] normalization)
    val_loader = build_google_vit_val_loader(
        args.data_dir, args.batch_size,
        image_size=args.image_size, num_workers=args.num_workers,
    )

    print(f"Evaluating at {args.image_size}x{args.image_size} on {len(val_loader.dataset)} images...")
    results = evaluate(model, val_loader, device, dtype)

    print(f"\n{'='*40}")
    print(f"  Top-1 Accuracy: {results['top1']:.2f}%")
    print(f"  Top-5 Accuracy: {results['top5']:.2f}%")
    print(f"  Total samples:  {results['total_samples']}")
    print(f"{'='*40}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
