"""AMP reward composition, independent of the simulator."""
import torch


def fuse_amp_rewards(task, style_raw, stand_mask, dones, task_weight, direction_gate=None):
    """Gate positive style bonuses; retain negative style costs in walking states."""
    if direction_gate is None:
        style_used = style_raw
        gate = torch.ones_like(style_raw)
    else:
        gate = direction_gate.clamp(0.0, 1.0)
        style_used = style_raw.clamp(max=0.0) + gate * style_raw.clamp(min=0.0)
        style_used = torch.where((~stand_mask) & (~dones.bool()), style_used,
                                 torch.zeros_like(style_used))
    task_contribution = torch.where(stand_mask, task, task_weight * task)
    style_contribution = torch.where(stand_mask, torch.zeros_like(style_used),
                                     (1.0 - task_weight) * style_used)
    return {
        "fused": task_contribution + style_contribution,
        "task_contribution": task_contribution,
        "style_contribution": style_contribution,
        "style_raw": style_raw,
        "style_used": style_used,
        "direction_gate": gate,
    }
