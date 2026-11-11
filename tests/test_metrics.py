import numpy as np
import pytest

from wisdypose.metrics import mpjpe, pa_mpjpe, pck


def _poses(n=5, j=17, d=2, seed=0):
    return np.random.default_rng(seed).normal(size=(n, j, d))


def test_mpjpe_zero_for_identical_poses():
    gt = _poses()
    assert mpjpe(gt, gt) == 0.0


def test_pa_mpjpe_ignores_similarity_transforms():
    gt = _poses()
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pred = 2.5 * gt @ rot + np.array([3.0, -1.0])
    assert mpjpe(pred, gt) > 1.0
    assert pa_mpjpe(pred, gt) < 1e-9


def test_pck_thresholds_relative_to_torso():
    gt = np.zeros((1, 17, 2))
    gt[0, 11] = [1.0, 0.0]                  # torso size 1
    pred = gt.copy()
    pred[0, 0] += [0.3, 0.0]                # one joint off by 0.3 torso
    assert pck(pred, gt, 20) == pytest.approx(100.0 * 16 / 17)
    assert pck(pred, gt, 30) == 100.0
