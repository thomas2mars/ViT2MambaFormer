import torch
import json
import sys
import os
import argparse
import numpy as np
from torchvision.models import vit_b_16
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.vit_utils import ViTStatesExtractor, convert_npz_to_torchvision
from utils.data import build_val_dataloader
from MambaFormer.MambaFormer import MambaFormer_Base_expand1_light_BiMamba2
from MambaFormer.utils import (
    compute_ssd_attention_map,
    patch_student_for_extraction,
)
from MO.config import DistillationConfig
from MO.losses import (
    combined_distillation_loss, cosine_similarity_metric, JS_divergence_metric,
)

# Clear registry
try:
    from torchvision.models._api import BUILTIN_MODELS
    BUILTIN_MODELS.clear()
except ImportError:
    pass


def load_models(cfg, checkpoint_path, device):
    # Teacher
    teacher = vit_b_16(weights=None, image_size=384)
    if cfg.teacher_weights_path.endswith('.npz'):
        teacher_state_dict = convert_npz_to_torchvision(
            cfg.teacher_weights_path, num_layers=12, embed_dim=768, num_heads=12
        )
    else:
        teacher_state_dict = torch.load(cfg.teacher_weights_path, map_location='cpu', weights_only=True)
    teacher.load_state_dict(teacher_state_dict)
    teacher.to(device)
    teacher.eval()

    extractor = ViTStatesExtractor(
        teacher, layer_indices=None, extract_attention=True,
        double_cls_token=cfg.double_cls_token
    )

    # Student
    student = MambaFormer_Base_expand1_light_BiMamba2(double_cls_token=cfg.double_cls_token, image_size=384)
    separate_directions = student.separate_directions

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    student.load_state_dict(checkpoint['model_state_dict'])
    student.to(device)
    student.eval()

    epoch = checkpoint.get('epoch', '?')
    print(f"Loaded checkpoint from epoch {epoch}")

    return teacher, extractor, student, separate_directions


def evaluate(student, teacher_extractor, get_layer_states, clear_states, val_loader, cfg, device, dtype):
    num_layers = cfg.num_layers

    # Per-layer accumulators
    layer_cos_sims = {i: [] for i in range(num_layers)}
    layer_js_divs = {i: [] for i in range(num_layers)}
    layer_losses = {i: [] for i in range(num_layers)}

    with torch.no_grad():
        interactive = not os.environ.get('SLURM_JOB_ID')
        for batch, _ in tqdm(val_loader, desc="Evaluating", disable=not interactive):
            batch = batch.to(device, non_blocking=True)
            clear_states()

            with torch.amp.autocast('cuda', enabled=True, dtype=dtype):
                _, t_states = teacher_extractor.get_vit_states(batch)

                for i in range(num_layers):
                    if i == 0:
                        inp = t_states['first_layer_input']
                    else:
                        inp = t_states['layers_output'][f'layer_{i-1}']
                    t_attn = t_states['attention_maps'][f'layer_{i}']

                    _ = student.encoder.layers[i](inp)
                    s_states = get_layer_states(i)
                    _, _, s_attn = compute_ssd_attention_map(
                        s_states, weighted=True, normalization='softmax',
                        per_head=True, temp=cfg.temp
                    )

                    loss = combined_distillation_loss(s_attn, t_attn, cfg.double_cls_token)
                    cos_sim = cosine_similarity_metric(s_attn, t_attn)
                    js_div = JS_divergence_metric(s_attn, t_attn)

                    layer_losses[i].append(loss.item())
                    layer_cos_sims[i].append(cos_sim)
                    layer_js_divs[i].append(js_div)

    # Aggregate results
    results = {}
    total_loss, total_cos, total_js = 0.0, 0.0, 0.0

    for i in range(num_layers):
        avg_loss = float(np.mean(layer_losses[i]))
        avg_cos = float(np.mean(layer_cos_sims[i]))
        avg_js = float(np.mean(layer_js_divs[i]))
        results[f'layer_{i}'] = {
            'loss': avg_loss,
            'cosine_similarity': avg_cos,
            'js_divergence': avg_js,
        }
        total_loss += avg_loss
        total_cos += avg_cos
        total_js += avg_js

    results['average'] = {
        'loss': total_loss / num_layers,
        'cosine_similarity': total_cos / num_layers,
        'js_divergence': total_js / num_layers,
    }

    return results


def print_results(results, num_layers):
    print(f"\n{'='*60}")
    print(f"{'Layer':<10} {'Loss':>10} {'Cos Sim':>10} {'JS Div':>10}")
    print(f"{'-'*60}")
    for i in range(num_layers):
        r = results[f'layer_{i}']
        print(f"{'Layer ' + str(i):<10} {r['loss']:>10.4f} {r['cosine_similarity']:>10.4f} {r['js_divergence']:>10.4f}")
    print(f"{'-'*60}")
    avg = results['average']
    print(f"{'AVERAGE':<10} {avg['loss']:>10.4f} {avg['cosine_similarity']:>10.4f} {avg['js_divergence']:>10.4f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (e.g. logs/distill_MO_base_10pct_v1/checkpoint_latest.pt)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to ImageNet dataset')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON (default: same dir as checkpoint)')
    args = parser.parse_args()

    cfg = DistillationConfig()
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir

    device = torch.device('cuda:0')
    dtype = torch.bfloat16
    torch.set_float32_matmul_precision("high")

    # Load models
    teacher, extractor, student, separate_directions = load_models(cfg, args.checkpoint, device)

    # Data (single GPU, val only)
    val_loader = build_val_dataloader(
        cfg.data_dir, cfg.base_batch_size, num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor, image_size=384, normalization='google_vit'
    )

    # Patch for state extraction
    unpatch, get_layer_states, clear_states = patch_student_for_extraction(
        student, cfg.num_layers, separate_directions=separate_directions
    )

    try:
        results = evaluate(student, extractor, get_layer_states, clear_states,
                           val_loader, cfg, device, dtype)
    finally:
        unpatch()

    print_results(results, cfg.num_layers)

    # Save JSON
    if args.output is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        output_path = os.path.join(checkpoint_dir, f'eval_{ckpt_name}.json')
    else:
        output_path = args.output

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
