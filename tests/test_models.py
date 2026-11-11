import glob

import pytest
import torch

from wisdypose.models import DenoisingAutoencoder, SDyConv2d, SubcarrierAttention
from wisdypose.pipeline import build_model, load_config

CONFIGS = sorted(glob.glob("configs/protocol*.yaml"))


@pytest.mark.parametrize("path", CONFIGS)
def test_paper_configs_build_and_run(path):
    cfg = load_config(path)
    model = build_model(cfg).eval()
    out = model(torch.rand(4, 3, 136, 32))
    assert out.shape == (4, 17, 2)
    n_stacks = len(cfg["estimator"]["n_kernels"])
    assert len(model.estimator.features) == n_stacks


def test_autoencoder_reconstructs_input_shape():
    ae = DenoisingAutoencoder((3, 136, 32), (36, 89))
    assert ae(torch.rand(2, 3, 136, 32)).shape == (2, 3, 136, 32)
    assert ae.code_shape == (89, 34, 8)


def test_attention_is_a_distribution_over_basis_kernels_per_subcarrier():
    att = SubcarrierAttention(8, n_basis=5, kernel_size=3, stride=1, padding=1, temperature=31).eval()
    alpha = att(torch.rand(2, 8, 34, 8))
    assert alpha.shape == (2, 5, 34)
    torch.testing.assert_close(alpha.sum(1), torch.ones(2, 34))


def test_sdyconv_equals_per_subcarrier_kernel_aggregation():
    torch.manual_seed(0)
    conv = SDyConv2d(4, 6, kernel_size=3, n_basis=3, temperature=1.0).eval()
    x = torch.rand(2, 4, 10, 5)
    alpha = conv.attention(x)                                  # [B, K, Nsc]
    y = conv(x)
    for b in range(2):
        for s in range(10):
            kernel = (alpha[b, :, s, None, None, None, None] * conv.weight).sum(0)   # Eq. (9)
            ref = torch.nn.functional.conv2d(x[b:b + 1], kernel, padding=1)[0, :, s]
            torch.testing.assert_close(y[b, :, s], ref, rtol=1e-4, atol=1e-5)


def test_denoiser_is_frozen_and_stays_in_eval_mode():
    cfg = load_config("configs/protocol1_noise10.yaml")
    model = build_model(cfg).train()
    assert not model.denoiser.training
    assert all(not p.requires_grad for p in model.denoiser.parameters())
    assert all(p.requires_grad for p in model.estimator.parameters())
