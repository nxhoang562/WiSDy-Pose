import os
import sys

import numpy as np
import pytest
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wisdypose.data import SUBJECTS, scene_of  # noqa: E402


@pytest.fixture(scope="session")
def fake_mmfi(tmp_path_factory):
    """A tiny MM-Fi-shaped tree: every subject, two actions, four frames each."""
    root = tmp_path_factory.mktemp("mmfi")
    rng = np.random.default_rng(0)
    for subject in SUBJECTS:
        for action in ("A02", "A03"):
            seq = root / scene_of(subject) / subject / action
            (seq / "wifi-csi").mkdir(parents=True)
            np.save(seq / "ground_truth.npy", rng.normal(size=(297, 17, 3)).astype(np.float32))
            for i in range(4):
                amp = rng.uniform(20, 60, size=(3, 114, 10))
                amp[0, 5, 3] = np.inf
                sio.savemat(seq / "wifi-csi" / f"frame{i + 1:03d}.mat", {"CSIamp": amp})
    return str(root)
