import torch
import torch.nn.functional as F


def frobenius_loss(student_attn, teacher_attn):
    return torch.norm(student_attn - teacher_attn, p='fro', dim=(-2, -1)).mean()


def kl_loss(student_attn, teacher_attn, eps=1e-8):
    student_probs = student_attn / (student_attn.sum(dim=-1, keepdim=True) + eps)
    student_probs = torch.clamp(student_probs, min=eps)
    student_log_probs = torch.log(student_probs)
    teacher_probs = torch.clamp(teacher_attn, min=eps)
    return F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1).mean()


def combined_distillation_loss(student_attn, teacher_attn, double_cls_token=True, eps=1e-8):
    """Fused loss computation - reduces memory allocations."""
    # Extract regions once
    if double_cls_token:
        cls_cols_s = student_attn[:, :, :, [0, -1]]
        cls_cols_t = teacher_attn[:, :, :, [0, -1]]
        cls_rows_s = student_attn[:, :, [0, -1], :]
        cls_rows_t = teacher_attn[:, :, [0, -1], :]
        patch_s = student_attn[:, :, 1:-1, 1:-1]
        patch_t = teacher_attn[:, :, 1:-1, 1:-1]
    else:
        cls_cols_s = student_attn[:, :, :, [0]]
        cls_cols_t = teacher_attn[:, :, :, [0]]
        cls_rows_s = student_attn[:, :, [0], :]
        cls_rows_t = teacher_attn[:, :, [0], :]
        patch_s = student_attn[:, :, 1:, 1:]
        patch_t = teacher_attn[:, :, 1:, 1:]

    # Frobenius losses (fused computation)
    fro_cls_cols = torch.norm(cls_cols_s - cls_cols_t, p='fro', dim=(-2, -1)).mean()
    fro_cls_rows = torch.norm(cls_rows_s - cls_rows_t, p='fro', dim=(-2, -1)).mean()
    fro_patches = torch.norm(patch_s - patch_t, p='fro', dim=(-2, -1)).mean()

    # KL divergence
    student_probs = student_attn / (student_attn.sum(dim=-1, keepdim=True) + eps)
    student_probs = torch.clamp(student_probs, min=eps)
    teacher_probs = torch.clamp(teacher_attn, min=eps)
    kl = F.kl_div(student_probs.log(), teacher_probs, reduction='none').sum(dim=-1).mean()

    # Weighted combination: 0.2/0.2/0.4/0.2
    loss = 0.2 * fro_cls_cols + 0.2 * fro_cls_rows + 0.4 * fro_patches + 0.2 * kl

    return loss


def cosine_similarity_metric(student_attn, teacher_attn):
    batch, heads = student_attn.shape[:2]
    s_flat = student_attn.view(batch, heads, -1)
    t_flat = teacher_attn.view(batch, heads, -1)
    return F.cosine_similarity(s_flat, t_flat, dim=-1).mean().item()


def JS_divergence_metric(student_attn, teacher_attn, eps=1e-8):
    s = student_attn / (student_attn.sum(dim=-1, keepdim=True) + eps)
    t = teacher_attn / (teacher_attn.sum(dim=-1, keepdim=True) + eps)
    s = torch.clamp(s, min=eps)
    t = torch.clamp(t, min=eps)
    m = 0.5 * (s + t)
    kl_t = F.kl_div(m.log(), t, reduction='none').sum(dim=-1)
    kl_s = F.kl_div(m.log(), s, reduction='none').sum(dim=-1)
    return 0.5 * (kl_t + kl_s).mean().item()
