import numpy as np
import torch

from wisdypose.data import (MMFiCSI, PROTOCOL_ACTIONS, build_datasets, resize_nearest, split_subjects,
                         val_test_indices)


def test_random_split_partitions_subjects_per_action():
    train, held_out = split_subjects("protocol1", "random", 0.8, 42)
    for action in PROTOCOL_ACTIONS["protocol1"]:
        tr = {s for s, a in train.items() if action in a}
        te = {s for s, a in held_out.items() if action in a}
        assert len(tr) == 32 and len(te) == 8 and not tr & te


def test_cross_splits():
    _, held = split_subjects("protocol2", "cross_subject")
    assert sorted(held) == ["S05", "S10", "S15", "S20", "S25", "S30", "S35", "S40"]
    train, held = split_subjects("protocol3", "cross_environment")
    assert sorted(held)[0] == "S31" and len(train) == 30


def test_val_test_split_matches_sklearn():
    from sklearn.model_selection import train_test_split
    n = 1001
    val, test = val_test_indices(n, 41)
    sk_val, sk_test = train_test_split(np.arange(n), test_size=0.5, random_state=41)
    assert np.array_equal(val, sk_val) and np.array_equal(test, sk_test)


def test_resize_nearest_index_rule():
    x = np.arange(3 * 114 * 10, dtype=float).reshape(3, 114, 10)
    y = resize_nearest(x, (136, 32))
    assert y.shape == (3, 136, 32)
    assert y[0, 135, 31] == x[0, 113, 9] and y[0, 0, 0] == x[0, 0, 0]


def test_dataset_items_are_normalised_and_eval_noise_is_fixed(fake_mmfi):
    cfg = {"data": {"protocol": "protocol1", "split": "random", "noise_level": 0.2}}
    train, val, test = build_datasets(cfg, fake_mmfi)
    assert len(train) + len(val) + len(test) == 40 * 2 * 4
    noisy, clean, kp = val[0]
    assert noisy.shape == clean.shape == (3, 136, 32) and kp.shape == (17, 2)
    assert torch.isfinite(noisy).all()
    for t in (noisy, clean):
        assert torch.allclose(t.amin(dim=(1, 2)), torch.zeros(3)) and torch.allclose(t.amax(dim=(1, 2)), torch.ones(3))
    assert torch.equal(val[0][0], noisy)                       # deterministic held-out noise
    assert not torch.equal(train[0][0], train[0][0])           # fresh training noise


def test_clean_setting_has_no_noise(fake_mmfi):
    form = {"S01": ["A02"]}
    ds = MMFiCSI(fake_mmfi, form, noise_level=0.0)
    noisy, clean, _ = ds[0]
    assert torch.equal(noisy, clean)
