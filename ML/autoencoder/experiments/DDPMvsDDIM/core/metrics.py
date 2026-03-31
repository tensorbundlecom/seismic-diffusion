import numpy as np
import torch
import torch.nn.functional as F


def spec_corr(target_spec: np.ndarray, pred_spec: np.ndarray) -> float:
    target_flat = target_spec.reshape(-1)
    pred_flat = pred_spec.reshape(-1)
    if np.std(target_flat) < 1e-12 or np.std(pred_flat) < 1e-12:
        return 0.0
    return float(np.corrcoef(target_flat, pred_flat)[0, 1])


def lsd(target_spec: np.ndarray, pred_spec: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sqrt(np.mean((np.log(target_spec + eps) - np.log(pred_spec + eps)) ** 2)))


def _avg_pool_spec(spec: np.ndarray, kernel_size: int) -> np.ndarray:
    tensor = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0)
    pooled = F.avg_pool2d(tensor, kernel_size=kernel_size, stride=kernel_size)
    return pooled.squeeze(0).squeeze(0).numpy()


def mr_lsd(target_spec: np.ndarray, pred_spec: np.ndarray, eps: float = 1e-8) -> float:
    scores = [lsd(target_spec, pred_spec, eps=eps)]
    for kernel_size in (2, 4):
        target_pooled = _avg_pool_spec(target_spec, kernel_size=kernel_size)
        pred_pooled = _avg_pool_spec(pred_spec, kernel_size=kernel_size)
        scores.append(lsd(target_pooled, pred_pooled, eps=eps))
    return float(np.mean(scores))
