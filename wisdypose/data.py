"""MM-Fi WiFi-CSI data: splits, frame loading, normalisation and noise injection.

Paper Sections V-A and V-B-1. Each sample is one CSI frame of shape
N_tx x N_rx x N_sc x T = 1 x 3 x 114 x 10, stored as ``CSIamp`` (3, 114, 10) in
``<root>/<scene>/<subject>/<action>/wifi-csi/frameXXX.mat``; the 3-D key-points of
the sequence are in ``<root>/<scene>/<subject>/<action>/ground_truth.npy``.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset, Subset

SUBJECTS = [f"S{i:02d}" for i in range(1, 41)]
ACTIONS = [f"A{i:02d}" for i in range(1, 28)]

# Protocol 1: 14 daily activities, Protocol 2: 13 rehabilitation exercises, Protocol 3: all 27.
PROTOCOL_ACTIONS = {
    "protocol1": ["A02", "A03", "A04", "A05", "A13", "A14", "A17", "A18", "A19", "A20", "A21",
                  "A22", "A23", "A27"],
    "protocol2": ["A01", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "A15", "A16", "A24",
                  "A25", "A26"],
    "protocol3": ACTIONS,
}
# S2 (cross-subject): 8 test subjects. S3 (cross-environment): train on E01-E03, test on E04.
CROSS_SUBJECT_TEST = ["S05", "S10", "S15", "S20", "S25", "S30", "S35", "S40"]
CROSS_ENVIRONMENT_TEST = SUBJECTS[30:]
FRAMES_PER_SEQUENCE = 297
N_KEYPOINTS = 17


def scene_of(subject: str) -> str:
    """MM-Fi records ten subjects per environment: S01-S10 in E01, ..., S31-S40 in E04."""
    return f"E{(int(subject[1:]) - 1) // 10 + 1:02d}"


def split_subjects(protocol: str, split: str, train_ratio: float = 0.8,
                   seed: int = 42) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Map subject -> actions for the training and the held-out part of a protocol.

    ``random`` (S1) draws, for every action separately, a random subject permutation
    (seed + action index) and keeps the first ``train_ratio`` of the subjects for
    training. ``cross_subject`` (S2) and ``cross_environment`` (S3) hold out fixed
    subjects.
    """
    actions = PROTOCOL_ACTIONS[protocol]
    train: Dict[str, List[str]] = {}
    held_out: Dict[str, List[str]] = {}
    if split == "random":
        n_train = int(np.floor(train_ratio * len(SUBJECTS)))
        for i, action in enumerate(actions):
            order = np.random.RandomState(seed + i).permutation(len(SUBJECTS))
            train_subjects = {SUBJECTS[j] for j in order[:n_train]}
            for subject in SUBJECTS:
                target = train if subject in train_subjects else held_out
                target.setdefault(subject, []).append(action)
    elif split in ("cross_subject", "cross_environment"):
        test_subjects = CROSS_SUBJECT_TEST if split == "cross_subject" else CROSS_ENVIRONMENT_TEST
        for subject in SUBJECTS:
            (held_out if subject in test_subjects else train)[subject] = list(actions)
    else:
        raise ValueError(f"unknown split {split!r}; use random, cross_subject or cross_environment")
    return train, held_out


def load_csi_frame(path: str) -> np.ndarray:
    """CSI amplitude (3, 114, 10): inf/NaN replaced by the mean of their packet, then
    min-max normalised to [0, 1] over the whole frame (Eq. 12)."""
    amp = sio.loadmat(path)["CSIamp"].astype(np.float64)
    amp[np.isinf(amp)] = np.nan
    for t in range(amp.shape[2]):
        packet = amp[:, :, t]
        missing = np.isnan(packet)
        if missing.any():
            packet[missing] = packet[~missing].mean()
    return (amp - amp.min()) / (amp.max() - amp.min())


def resize_nearest(x: np.ndarray, size: Sequence[int]) -> np.ndarray:
    """Nearest-neighbour resize of a (C, H, W) array to (C, *size).

    Source index = floor(dst_index * src_len / dst_len), the rule used by OpenCV's
    INTER_NEAREST. The default (136, 32) maps 114 subcarriers x 10 packets onto a grid
    that the two 2x2 poolings of the Denoiser divide evenly.
    """
    h, w = size
    rows = np.minimum(np.floor(np.arange(h) * (x.shape[1] / h)).astype(int), x.shape[1] - 1)
    cols = np.minimum(np.floor(np.arange(w) * (x.shape[2] / w)).astype(int), x.shape[2] - 1)
    return x[:, rows][:, :, cols]


def minmax_per_channel(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalise each antenna channel of a (C, N_sc, T) tensor to [0, 1]."""
    lo = x.amin(dim=(1, 2), keepdim=True)
    hi = x.amax(dim=(1, 2), keepdim=True)
    return (x - lo) / (hi - lo).clamp_min(1e-12)


class MMFiCSI(Dataset):
    """CSI frames of one split. Item: (noisy input, clean target, key-points).

    noisy  = per-channel min-max of (frame + noise_level * N(0, 1))   -> network input
    clean  = per-channel min-max of frame                              -> Denoiser target
    key-points = first ``keypoint_dims`` coordinates of the 17 ground-truth joints.

    Training samples draw fresh noise at every access; with ``deterministic_noise`` the
    noise of item i is fixed by ``noise_seed + i`` so that validation and test sets are
    identical across epochs and runs.
    """

    def __init__(self, root: str, form: Dict[str, List[str]], resize: Sequence[int] = (136, 32),
                 noise_level: float = 0.0, keypoint_dims: int = 2,
                 deterministic_noise: bool = False, noise_seed: int = 0):
        self.root = root
        self.resize = tuple(resize)
        self.noise_level = float(noise_level)
        self.keypoint_dims = keypoint_dims
        self.deterministic_noise = deterministic_noise
        self.noise_seed = noise_seed
        self.items: List[Tuple[str, str, int]] = []   # (frame path, ground-truth path, frame index)
        for subject, actions in form.items():
            for action in actions:
                seq = os.path.join(root, scene_of(subject), subject, action)
                gt = os.path.join(seq, "ground_truth.npy")
                for idx in range(FRAMES_PER_SEQUENCE):
                    frame = os.path.join(seq, "wifi-csi", f"frame{idx + 1:03d}.mat")
                    if os.path.isfile(frame) and os.path.getsize(frame) > 0:
                        self.items.append((frame, gt, idx))
        if not self.items:
            raise FileNotFoundError(f"no MM-Fi WiFi-CSI frames found under {root!r}")

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    @lru_cache(maxsize=256)
    def _keypoints(gt_path: str) -> np.ndarray:
        return np.load(gt_path).astype(np.float32)

    def __getitem__(self, i: int):
        frame, gt, idx = self.items[i]
        clean = torch.from_numpy(resize_nearest(load_csi_frame(frame), self.resize)).float()
        noisy = clean
        if self.noise_level > 0:
            gen = torch.Generator().manual_seed(self.noise_seed + i) if self.deterministic_noise else None
            noisy = clean + self.noise_level * torch.randn(clean.shape, generator=gen)
        keypoints = torch.from_numpy(self._keypoints(gt)[idx, :, :self.keypoint_dims].copy())
        return minmax_per_channel(noisy), minmax_per_channel(clean), keypoints


def val_test_indices(n: int, seed: int = 41) -> Tuple[np.ndarray, np.ndarray]:
    """Split the held-out frames in half: a random permutation, the first ceil(n/2)
    frames form the test set and the rest the validation set."""
    order = np.random.RandomState(seed).permutation(n)
    n_test = math.ceil(0.5 * n)
    return order[n_test:], order[:n_test]


def build_datasets(cfg: dict, root: str):
    """Training, validation and test datasets described by the ``data`` config block."""
    d = cfg["data"]
    train_form, held_out_form = split_subjects(d["protocol"], d["split"], d.get("train_ratio", 0.8),
                                               d.get("split_seed", 42))
    common = dict(resize=d.get("resize", (136, 32)), noise_level=d.get("noise_level", 0.0),
                  keypoint_dims=d.get("keypoint_dims", 2))
    train = MMFiCSI(root, train_form, **common)
    held_out = MMFiCSI(root, held_out_form, deterministic_noise=True,
                       noise_seed=d.get("noise_seed", 0), **common)
    val_idx, test_idx = val_test_indices(len(held_out), d.get("val_test_seed", 41))
    return train, Subset(held_out, val_idx.tolist()), Subset(held_out, test_idx.tolist())


def make_loader(dataset, batch_size: int, train: bool, num_workers: int = 4, seed: int = 0):
    gen = torch.Generator().manual_seed(seed) if train else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train,
                      num_workers=num_workers, generator=gen, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)
