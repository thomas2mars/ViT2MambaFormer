"""
visualize_attention.py

Samples N random images from ImageNet val and produces one PNG per image showing:
  - The original image
  - ViT-B/16 teacher attention map per layer (average over heads, CLS→patches)
  - MambaFormer student attention map per layer (average over heads, CLS→patches)

The student checkpoint can come from any distillation step (MO, HA, or WK).
For MO/HA checkpoints, the student is fed teacher layer inputs (layer-by-layer) to
produce the most meaningful attention maps. For WK, same approach is used so the
comparison is always "same input → ViT vs Mamba attention".

Layout per PNG: 12 rows (one per layer) × 3 columns
  col 0 : original image (same across all rows, used as spatial reference)
  col 1 : ViT attention heatmap overlaid on image
  col 2 : MambaFormer attention heatmap overlaid on image

Usage:
    python tools/visualize_attention.py \\
        --checkpoint  path/to/checkpoint.pt \\
        --data_dir    path/to/ImageNet \\
        --output_dir  tools/attention_viz/run1 \\
        --num_samples 10 \\
        --seed        0
"""

import argparse
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import vit_b_16

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vit_utils import ViTStatesExtractor, convert_npz_to_torchvision
from MambaFormer.MambaFormer import MambaFormer_Base_expand1_light_BiMamba2
from MambaFormer.utils import compute_ssd_attention_map, patch_student_for_extraction

try:
    from torchvision.models._api import BUILTIN_MODELS
    BUILTIN_MODELS.clear()
except ImportError:
    pass

NUM_LAYERS = 12
GRID_SIZE = 24       # 384 / 16 = 24 patches per side
DOUBLE_CLS = True    # always True for this project


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_teacher(weights_path, device):
    teacher = vit_b_16(weights=None, image_size=384)
    if weights_path.endswith('.npz'):
        sd = convert_npz_to_torchvision(weights_path, num_layers=12, embed_dim=768, num_heads=12)
    else:
        sd = torch.load(weights_path, map_location='cpu', weights_only=True)
    teacher.load_state_dict(sd)
    teacher.to(device).eval()
    extractor = ViTStatesExtractor(
        teacher, layer_indices=None, extract_attention=True,
        average_attn_weights=False, double_cls_token=DOUBLE_CLS
    )
    return teacher, extractor


def load_student(checkpoint_path, device):
    student = MambaFormer_Base_expand1_light_BiMamba2(
        double_cls_token=DOUBLE_CLS, image_size=384
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'
    student.load_state_dict(sd)
    student.to(device).eval()
    print(f"Loaded student from {checkpoint_path} (epoch {epoch})")
    return student


# ─────────────────────────────────────────────
# Attention map → spatial heatmap
# ─────────────────────────────────────────────

def attn_to_heatmap(attn_map):
    """
    Convert a per-head attention map to a 2D spatial heatmap.

    Args:
        attn_map: Tensor [heads, seq_len, seq_len]
                  seq_len = 578 with double CLS (CLS@0, patches@1:577, CLS@577)

    Returns:
        np.ndarray [GRID_SIZE, GRID_SIZE] normalised to [0, 1]
    """
    # Average over heads → [seq_len, seq_len]
    avg = attn_map.float().mean(dim=0)       # [S, S]

    # CLS token (row 0) attention to each patch (cols 1:577)
    cls_row = avg[0, 1:-1]                   # [576]

    # Reshape to spatial grid
    spatial = cls_row.reshape(GRID_SIZE, GRID_SIZE).cpu().numpy()

    # Normalise to [0, 1]
    vmin, vmax = spatial.min(), spatial.max()
    if vmax > vmin:
        spatial = (spatial - vmin) / (vmax - vmin)
    return spatial


def upsample_heatmap(heatmap_np, target_size=384):
    """Bilinear upsample a [H, W] numpy heatmap to [target_size, target_size]."""
    t = torch.from_numpy(heatmap_np).unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(t, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


def compute_heatmap_metrics(h1, h2):
    """
    Compute JS Divergence and Cosine Similarity between two flattened heatmaps.

    Args:
        h1, h2: np.ndarray [H, W], values in [0, 1]

    Returns:
        jsd  (float): Jensen-Shannon divergence in [0, 1]
        cos  (float): Cosine similarity in [-1, 1]
    """
    eps = 1e-10
    p = h1.flatten().astype(np.float64) + eps
    q = h2.flatten().astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)
    jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))

    dot = np.dot(p, q)
    cos = dot / (np.linalg.norm(p) * np.linalg.norm(q) + eps)

    return float(jsd), float(cos)


# ─────────────────────────────────────────────
# Figure creation
# ─────────────────────────────────────────────

def make_figure(image_np, vit_heatmaps, mamba_heatmaps, class_name, sample_idx, output_path,
                layer_metrics=None):
    """
    Create and save one PNG for a single image.

    Layout:
      - Top banner:  original image (centered, full width)
      - 12 rows × 3 columns: [ViT layer i] [Mamba layer i] [metrics]

    Args:
        layer_metrics: dict {layer_idx: (jsd, cos)} — if provided, a third column
                       shows JS Divergence and Cosine Similarity per layer.
    """
    col_labels = ['ViT', 'MambaFormer']
    n_rows = NUM_LAYERS
    n_cols = 3  # ViT | Mamba | Metrics

    fig = plt.figure(figsize=(n_cols * 4 + 2, n_rows * 3 + 3))
    gs = gridspec.GridSpec(
        n_rows + 2, n_cols + 1,           # +1 row for header, +1 row for col labels, +1 col for row labels
        figure=fig,
        hspace=0.05, wspace=0.05,
        left=0.08, right=0.98, top=0.90, bottom=0.02,
        width_ratios=[0.3, 1, 1, 0.8],
        height_ratios=[2.8, 0.18] + [1] * n_rows,
    )

    fig.suptitle(
        f'Sample {sample_idx}  —  class: {class_name}',
        fontsize=13, fontweight='bold', y=0.93
    )

    # ── Row 0: original image centered across all columns ──
    ax_orig = fig.add_subplot(gs[0, 1:4])
    ax_orig.imshow(image_np, aspect='equal')
    ax_orig.axis('off')

    # ── Row 1: column labels ──
    for col_idx, label in enumerate(['ViT', 'MambaFormer']):
        ax_lbl = fig.add_subplot(gs[1, col_idx + 1])
        ax_lbl.axis('off')
        ax_lbl.text(0.5, 0.5, label, ha='center', va='center',
                    fontsize=13, fontweight='bold', transform=ax_lbl.transAxes)
    ax_mlbl = fig.add_subplot(gs[1, 3])
    ax_mlbl.axis('off')
    ax_mlbl.text(0.5, 0.5, 'JSD / Cos', ha='center', va='center',
                 fontsize=13, fontweight='bold', transform=ax_mlbl.transAxes)

    # ── Layer rows (start at row 2) ──
    for layer_idx in range(NUM_LAYERS):
        row = layer_idx + 2

        # Row label
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.axis('off')
        ax_label.text(0.95, 0.5, f'L{layer_idx}',
                      ha='right', va='center', fontsize=13,
                      transform=ax_label.transAxes)

        heatmap_up_vit   = upsample_heatmap(vit_heatmaps[layer_idx])
        heatmap_up_mamba = upsample_heatmap(mamba_heatmaps[layer_idx])

        for col_idx, (heatmap_up, label) in enumerate(
            [(heatmap_up_vit, 'ViT'), (heatmap_up_mamba, 'MambaFormer')]
        ):
            ax = fig.add_subplot(gs[row, col_idx + 1])
            ax.imshow(image_np)
            ax.imshow(heatmap_up, alpha=0.55, cmap='jet', vmin=0, vmax=1)
            ax.axis('off')

        # Metrics column
        ax_m = fig.add_subplot(gs[row, 3])  # row already offset by 2
        ax_m.axis('off')
        if layer_metrics is not None and layer_idx in layer_metrics:
            jsd, cos = layer_metrics[layer_idx]
            metric_text = f"JSD:  {jsd:.4f}\nCos:  {cos:.4f}"
        else:
            metric_text = ""
        ax_m.text(0.5, 0.5, metric_text,
                  ha='center', va='center', fontsize=16,
                  fontfamily='monospace',
                  transform=ax_m.transAxes)

    plt.savefig(output_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize ViT vs MambaFormer attention maps')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to MambaFormer checkpoint (MO, HA, or WK step)')
    parser.add_argument('--teacher_weights', type=str, default='models/vit/ViT-B_16.npz',
                        help='Path to ViT-B/16 weights (.npz or .pt)')
    parser.add_argument('--data_dir', type=str, default='dataset/ImageNet_ILSVRC2012',
                        help='Path to ImageNet dataset root')
    parser.add_argument('--output_dir', type=str, default='tools/attention_viz',
                        help='Directory where PNGs are saved')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of images to visualise')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for image sampling')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    print(f"Device: {device}")

    # ── Load models ──
    print("Loading teacher (ViT-B/16)...")
    _, extractor = load_teacher(args.teacher_weights, device)

    print("Loading student (MambaFormer)...")
    student = load_student(args.checkpoint, device)

    # ── Dataset ──
    val_transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'val'), transform=val_transform
    )
    class_names = val_dataset.classes

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(val_dataset)), args.num_samples)
    print(f"Sampled {args.num_samples} images (seed={args.seed})")

    # ── Patch student for state extraction ──
    unpatch, get_layer_states, clear_states = patch_student_for_extraction(
        student, NUM_LAYERS, separate_directions=student.separate_directions
    )

    try:
        for sample_num, idx in enumerate(indices):
            image_tensor, label = val_dataset[idx]
            class_name = class_names[label]
            print(f"[{sample_num+1}/{args.num_samples}] idx={idx}  class={class_name}")

            batch = image_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                autocast_ctx = torch.amp.autocast('cuda', dtype=dtype) if device.type == 'cuda' \
                               else torch.amp.autocast('cpu', dtype=torch.float32)
                with autocast_ctx:
                    # Teacher states (attention maps + per-layer inputs/outputs)
                    _, t_states = extractor.get_vit_states(batch)

                    # Student attention maps: feed teacher layer inputs to student layers
                    s_attn_maps = {}
                    for layer_idx in range(NUM_LAYERS):
                        clear_states()
                        inp = (t_states['first_layer_input'] if layer_idx == 0
                               else t_states['layers_output'][f'layer_{layer_idx - 1}'])
                        _ = student.encoder.layers[layer_idx](inp)
                        layer_states = get_layer_states(layer_idx)
                        _, _, s_attn = compute_ssd_attention_map(
                            layer_states, weighted=True, normalization='softmax',
                            per_head=True, temp=1.0
                        )
                        s_attn_maps[layer_idx] = s_attn[0].cpu()   # [heads, S, S]

            # ── Build per-layer heatmaps and metrics ──
            vit_heatmaps   = {}
            mamba_heatmaps = {}
            layer_metrics  = {}
            for layer_idx in range(NUM_LAYERS):
                t_attn = t_states['attention_maps'][f'layer_{layer_idx}'][0].cpu()  # [heads, S, S]
                vit_heatmaps[layer_idx]   = attn_to_heatmap(t_attn)
                mamba_heatmaps[layer_idx] = attn_to_heatmap(s_attn_maps[layer_idx])
                jsd, cos = compute_heatmap_metrics(
                    vit_heatmaps[layer_idx], mamba_heatmaps[layer_idx]
                )
                layer_metrics[layer_idx] = (jsd, cos)
                print(f"    L{layer_idx:02d}  JSD={jsd:.4f}  Cos={cos:.4f}")

            # Denormalize image for display (Google ViT: mean=0.5, std=0.5 → [0,1])
            image_np = (image_tensor * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()

            output_path = os.path.join(args.output_dir, f'sample_{sample_num:03d}_{class_name}.png')
            make_figure(image_np, vit_heatmaps, mamba_heatmaps, class_name, sample_num,
                        output_path, layer_metrics=layer_metrics)

    finally:
        unpatch()

    print(f"\nDone — {args.num_samples} PNGs saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
