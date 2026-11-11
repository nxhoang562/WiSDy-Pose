"""MPJPE, PA-MPJPE and PCK@alpha (Section V-B-2). Inputs are [N, J, D] arrays."""
from __future__ import annotations

import numpy as np

# Torso size of Eq. (13): distance between the right shoulder and the left hip.
TORSO_KEYPOINTS = (1, 11)


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-joint Euclidean error, in the unit of the inputs."""
    return float(np.linalg.norm(pred - gt, axis=-1).mean())


def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Align one pose ``pred`` [J, D] to ``gt`` with the optimal similarity transform
    (rotation, uniform scale, translation)."""
    mu_p, mu_g = pred.mean(0), gt.mean(0)
    p0, g0 = pred - mu_p, gt - mu_g
    norm_p, norm_g = np.linalg.norm(p0), np.linalg.norm(g0)
    p0, g0 = p0 / norm_p, g0 / norm_g
    u, s, vt = np.linalg.svd(g0.T @ p0)
    v = vt.T
    rot = v @ u.T
    if np.linalg.det(rot) < 0:      # reflection -> closest proper rotation
        v[:, -1] *= -1
        s[-1] *= -1
        rot = v @ u.T
    scale = s.sum() * norm_g / norm_p
    return scale * (pred - mu_p) @ rot + mu_g


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """MPJPE after Procrustes alignment of every pose."""
    aligned = np.stack([procrustes_align(p, g) for p, g in zip(pred, gt)])
    return mpjpe(aligned, gt)


def pck(pred: np.ndarray, gt: np.ndarray, alpha: float) -> float:
    """PCK@alpha (Eq. 13), in %: share of joints whose error is at most alpha/100 times
    the torso size of the ground-truth pose."""
    a, b = TORSO_KEYPOINTS
    torso = np.linalg.norm(gt[:, a] - gt[:, b], axis=-1)
    dist = np.linalg.norm(pred - gt, axis=-1) / torso[:, None]
    return float(100.0 * np.mean(dist <= alpha / 100.0))


def pose_metrics(pred: np.ndarray, gt: np.ndarray, to_mm: float = 1000.0,
                 alphas=(20, 30, 40, 50)) -> dict:
    """All paper metrics; MM-Fi key-points are in metres, errors are reported in mm."""
    out = {"mpjpe_mm": mpjpe(pred, gt) * to_mm, "pa_mpjpe_mm": pa_mpjpe(pred, gt) * to_mm}
    out.update({f"pck@{a}": pck(pred, gt, a) for a in alphas})
    return out
