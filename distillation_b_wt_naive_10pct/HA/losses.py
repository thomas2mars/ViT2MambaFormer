import torch
import torch.nn.functional as F


def l2_loss(student_output, teacher_output):
    return F.mse_loss(student_output, teacher_output, reduction='mean')


def frobenius_loss(student_attn, teacher_attn):
    return torch.norm(student_attn - teacher_attn, p='fro', dim=(-2, -1)).mean()


def combined_output_attention_loss(student_output, teacher_output, student_attn, teacher_attn):
    """Combined loss: 0.7 * L2_output + 0.3 * Frobenius_attention."""
    loss_l2 = l2_loss(student_output, teacher_output)
    loss_fro = frobenius_loss(student_attn, teacher_attn)
    total_loss = 0.7 * loss_l2 + 0.3 * loss_fro
    return total_loss, loss_l2, loss_fro


def linear_cka(student_out, teacher_out, eps=1e-12):
    """Linear Centered Kernel Alignment between student and teacher outputs."""
    Xc = student_out - student_out.mean(dim=1, keepdim=True)
    Yc = teacher_out - teacher_out.mean(dim=1, keepdim=True)
    XtX = torch.matmul(Xc.transpose(1, 2), Xc)
    YtY = torch.matmul(Yc.transpose(1, 2), Yc)
    YtX = torch.matmul(Yc.transpose(1, 2), Xc)
    num = (YtX.pow(2)).sum(dim=(1, 2))
    denom = (XtX.pow(2)).sum(dim=(1, 2)).sqrt() * (YtY.pow(2)).sum(dim=(1, 2)).sqrt() + eps
    cka = num / denom
    return cka.mean(dim=0)
