import torch
import torch.nn.functional as F


def logit_distillation_loss(student_logits, teacher_logits, labels,
                            temperature=1.0, kl_weight=0.7, ce_weight=0.3,
                            label_smoothing=0.0):
    """Combined KL divergence on soft targets + CE on hard labels."""
    # KL on softened logits
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    kl_loss = kl_loss * (temperature ** 2)

    # CE on hard labels with label smoothing
    ce_loss = F.cross_entropy(student_logits, labels, label_smoothing=label_smoothing)

    return kl_weight * kl_loss + ce_weight * ce_loss


def hard_distillation_loss(student_logits, teacher_logits, labels,
                           labels_b=None, lam=1.0,
                           alpha=0.5, label_smoothing=0.1):
    """DeiT hard-label distillation: CE with true labels + CE with teacher's hard prediction.

    When Mixup/CutMix is active, labels_b and lam are used to compute
    the mixed CE: lam * CE(s, y_a) + (1-lam) * CE(s, y_b).
    """
    # CE with teacher's hard decision (argmax)
    teacher_hard = teacher_logits.argmax(dim=1)
    ce_teacher = F.cross_entropy(student_logits, teacher_hard)

    # CE with true labels (handles Mixup/CutMix via label mixing)
    if labels_b is not None:
        ce_true = lam * F.cross_entropy(student_logits, labels, label_smoothing=label_smoothing) + \
                  (1 - lam) * F.cross_entropy(student_logits, labels_b, label_smoothing=label_smoothing)
    else:
        ce_true = F.cross_entropy(student_logits, labels, label_smoothing=label_smoothing)

    return (1 - alpha) * ce_true + alpha * ce_teacher


def compute_agreement(student_logits, teacher_logits):
    """Top-1 agreement: percentage of matching predictions between student and teacher."""
    _, student_pred = torch.max(student_logits.data, 1)
    _, teacher_pred = torch.max(teacher_logits.data, 1)
    total = teacher_logits.size(0)
    correct = (student_pred == teacher_pred).sum().item()
    return correct, total
