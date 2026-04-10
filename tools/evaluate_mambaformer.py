"""
ImageNet Top-1 / Top-5 accuracy evaluation for the distilled MambaFormer student.

Usage:
    python Distillation_WK/evaluate.py --checkpoint path/to/checkpoint.pt --data_dir path/to/ImageNet
"""
import torch
import json
import sys
import os
import argparse
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data import build_val_dataloader
from MambaFormer.MambaFormer import MambaFormer_Base_expand1_light_BiMamba2

# Clear registry
try:
    from torchvision.models._api import BUILTIN_MODELS
    BUILTIN_MODELS.clear()
except ImportError:
    pass


def load_student(checkpoint_path, double_cls_token, image_size, device):
    student = MambaFormer_Base_expand1_light_BiMamba2(double_cls_token=double_cls_token, image_size=image_size)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    student.load_state_dict(checkpoint['model_state_dict'])
    student.to(device)
    student.eval()
    epoch = checkpoint.get('epoch', '?')
    print(f"Loaded checkpoint from epoch {epoch}")
    return student


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

        # Top-1
        pred_1 = logits.argmax(dim=1)
        correct_1 += (pred_1 == targets).sum().item()

        # Top-5
        _, pred_5 = logits.topk(5, dim=1)
        correct_5 += (pred_5 == targets.unsqueeze(1)).any(dim=1).sum().item()

        total += targets.size(0)

    top1 = 100.0 * correct_1 / total
    top5 = 100.0 * correct_5 / total
    return {'top1': top1, 'top5': top5, 'total_samples': total}


def main():
    parser = argparse.ArgumentParser(description='ImageNet Top-1/Top-5 evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='dataset/ImageNet_ILSVRC2012')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=384,
                        help='Evaluation resolution (384 for fine-tuned ViT, 224 for base)')
    parser.add_argument('--double_cls_token', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device('cuda:0')
    dtype = torch.bfloat16
    torch.set_float32_matmul_precision("high")

    student = load_student(args.checkpoint, args.double_cls_token, args.image_size, device)

    val_loader = build_val_dataloader(
        args.data_dir, args.batch_size, num_workers=args.num_workers, prefetch_factor=3,
        image_size=args.image_size, normalization='google_vit'
    )

    print(f"Evaluating at {args.image_size}x{args.image_size} resolution on {len(val_loader.dataset)} images...")
    results = evaluate(student, val_loader, device, dtype)

    print(f"\n{'='*40}")
    print(f"  Top-1 Accuracy: {results['top1']:.2f}%")
    print(f"  Top-5 Accuracy: {results['top5']:.2f}%")
    print(f"  Total samples:  {results['total_samples']}")
    print(f"{'='*40}")

    # Save JSON
    if args.output is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        output_path = os.path.join(checkpoint_dir, f'eval_top1_{ckpt_name}.json')
    else:
        output_path = args.output

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
