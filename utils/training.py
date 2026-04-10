import os
import glob
import json
import torch
import torch.distributed as dist
import numpy as np


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        return local_rank, rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(student_ref, optimizer, scheduler, epoch, all_epoch_metrics, root_dir):
    """Save full training state for robust resumption."""
    path = os.path.join(root_dir, f"checkpoint_ep{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': student_ref.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': all_epoch_metrics,
    }, path)

    # Also save a symlink/copy to "latest" for easy resume
    latest_path = os.path.join(root_dir, "checkpoint_latest.pt")
    # Atomic replace: write to temp then rename
    tmp_path = latest_path + ".tmp"
    torch.save({
        'epoch': epoch,
        'model_state_dict': student_ref.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': all_epoch_metrics,
    }, tmp_path)
    os.replace(tmp_path, latest_path)

    return path


def load_checkpoint(root_dir, student_ref, optimizer, scheduler, device):
    """Load the latest checkpoint if it exists.

    Returns:
        (start_epoch, all_epoch_metrics) if checkpoint found, else (0, [])
    """
    latest_path = os.path.join(root_dir, "checkpoint_latest.pt")
    if not os.path.exists(latest_path):
        return 0, []

    checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
    student_ref.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch']
    all_epoch_metrics = checkpoint.get('metrics', [])

    return start_epoch, all_epoch_metrics


def aggregate_layer_metrics(layer_data, num_layers, metric_names=None):
    """Compute per-layer and average metrics from accumulated lists.

    Args:
        layer_data: dict mapping metric_name -> {layer_key -> list of values}.
                    The first key is treated as the primary 'loss' metric.
        num_layers: number of layers.
        metric_names: optional dict mapping internal keys to output names.
                      e.g. {'losses': 'loss', 'cos_sims': 'cosine_similarity'}
                      If None, defaults to MO-style names.

    Returns:
        metrics dict with per-layer and average entries.
    """
    if metric_names is None:
        metric_names = {
            'losses': 'loss',
            'cos_sims': 'cosine_similarity',
            'js_divs': 'js_divergence',
        }

    metrics = {}
    accumulators = {out_name: 0.0 for out_name in metric_names.values()}
    layers_with_metrics = 0

    # Use the first metric key to check if data exists for a layer
    primary_key = next(iter(metric_names))

    for i in range(num_layers):
        layer_key = f'layer_{i}'
        if layer_data[primary_key][layer_key]:
            layer_metrics = {}
            for data_key, out_name in metric_names.items():
                values = layer_data[data_key][layer_key]
                avg = float(np.mean(values)) if values else 0.0
                layer_metrics[out_name] = avg
                accumulators[out_name] += avg

            metrics[layer_key] = layer_metrics
            layers_with_metrics += 1

    if layers_with_metrics > 0:
        metrics['average'] = {
            out_name: float(total / layers_with_metrics)
            for out_name, total in accumulators.items()
        }

    return metrics


def save_metrics(all_epoch_metrics, root_dir):
    metrics_file = os.path.join(root_dir, 'training_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(all_epoch_metrics, f, indent=2)
    return metrics_file


def init_layer_trackers(num_layers, keys=None):
    """Create empty per-layer metric accumulators.

    Args:
        num_layers: number of layers.
        keys: list of metric keys. Defaults to MO-style ['losses', 'cos_sims', 'js_divs'].
    """
    if keys is None:
        keys = ['losses', 'cos_sims', 'js_divs']
    return {key: {f'layer_{i}': [] for i in range(num_layers)} for key in keys}
