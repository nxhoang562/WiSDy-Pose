"""Denoiser (autoencoder, Section IV-B) and Estimator (stacked SDy-CNN, Section IV-C).

Tensors follow the paper's layout [batch, N_c, N_sc, T]: antenna channels, subcarriers,
CSI packets.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def conv_bn_relu(c_in: int, c_out: int, k: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, k, padding=k // 2), nn.BatchNorm2d(c_out), nn.ReLU())


class DenoisingAutoencoder(nn.Module):
    """Fig. 5: encoder = 2 x (Conv2D + BN + ReLU, MaxPooling2D); decoder mirrors it with
    UpSampling2D and ends with a Conv2D + BN + ReLU output layer. After stage 1 only the
    encoder is kept as the Denoiser; its output is the code C_i of shape N_c' x N_sc' x T'."""

    def __init__(self, in_shape: Sequence[int] = (3, 136, 32), n_kernels: Sequence[int] = (36, 89),
                 kernel_sizes: Sequence[int] = (3, 3), pool_sizes: Sequence[int] = (2, 2)):
        super().__init__()
        (n1, n2), (k1, k2), (p1, p2) = n_kernels, kernel_sizes, pool_sizes
        c = in_shape[0]
        self.in_shape = tuple(in_shape)
        self.encoder = nn.Sequential(conv_bn_relu(c, n1, k1), nn.MaxPool2d(p1),
                                     conv_bn_relu(n1, n2, k2), nn.MaxPool2d(p2))
        with torch.no_grad():   # spatial sizes before each pooling, restored by the decoder
            x = torch.zeros(1, *in_shape)
            size1 = tuple(self.encoder[0](x).shape[-2:])
            size2 = tuple(self.encoder[2](self.encoder[1](self.encoder[0](x))).shape[-2:])
            self.code_shape = tuple(self.encoder(x).shape[1:])
        self.decoder = nn.Sequential(conv_bn_relu(n2, n2, k2), nn.Upsample(size=size2),
                                     conv_bn_relu(n2, n1, k1), nn.Upsample(size=size1),
                                     conv_bn_relu(n1, c, k1))
        with torch.no_grad():
            out = tuple(self.decoder(self.encoder(torch.zeros(1, *in_shape))).shape[1:])
        if out != self.in_shape:
            raise ValueError(f"reconstruction shape {out} differs from input {self.in_shape}; "
                             "use odd kernel sizes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class SubcarrierAttention(nn.Module):
    """Attention over the K basis kernels for every subcarrier position.

    Eq. (6) squeeze: average over time -> Z [B, N_c, N_sc].
    Eq. (7) beta = ReLU(BN(Conv1D(W1, Z))), Conv1D along the subcarrier axis.
    Eq. (8) alpha = Softmax(Conv1D(W2, beta) / temperature) over the K kernels.
    The first Conv1D uses the kernel size, stride and padding of the 2-D convolution, so
    alpha has one entry per output subcarrier.
    """

    def __init__(self, in_ch: int, n_basis: int, kernel_size: int, stride: int, padding: int,
                 temperature: float):
        super().__init__()
        hidden = max(4, in_ch // 4)
        self.temperature = temperature
        self.conv1 = nn.Conv1d(in_ch, hidden, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, n_basis, 1)
        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.mean(dim=3)
        beta = F.relu(self.bn(self.conv1(z)))
        return F.softmax(self.conv2(beta) / self.temperature, dim=1)   # [B, K, N_sc']


class SDyConv2d(nn.Module):
    """Subcarrier-focusing dynamic convolution, Eqs. (9)-(10).

    The kernel applied at subcarrier s is kappa_s = sum_k alpha_k(s) W_k. Because
    convolution is linear in the kernel, this equals the alpha-weighted sum of the K
    basis-kernel outputs, which is how it is computed here (one grouped conv2d)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, n_basis: int, temperature: float,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.out_ch, self.n_basis, self.stride, self.padding = out_ch, n_basis, stride, padding
        self.weight = nn.Parameter(torch.empty(n_basis, out_ch, in_ch, kernel_size, kernel_size))
        for k in range(n_basis):
            nn.init.kaiming_normal_(self.weight[k])
        self.attention = SubcarrierAttention(in_ch, n_basis, kernel_size, stride, padding, temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.attention(x)
        y = F.conv2d(x, self.weight.flatten(0, 1), stride=self.stride, padding=self.padding)
        y = y.view(x.shape[0], self.n_basis, self.out_ch, *y.shape[-2:])
        return (y * alpha[:, :, None, :, None]).sum(dim=1)


class SDyCNNBlock(nn.Sequential):
    """One SDy-CNN network of Fig. 6: SDyConv2d, BN, ReLU, dropout, max-pooling."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, pool_size: int, n_basis: int,
                 temperature: float, dropout: float):
        super().__init__(SDyConv2d(in_ch, out_ch, kernel_size, n_basis, temperature),
                         nn.BatchNorm2d(out_ch), nn.ReLU(), nn.Dropout(dropout), nn.MaxPool2d(pool_size))


class Estimator(nn.Module):
    """Fig. 6: n_stacks = len(n_kernels) stacked SDy-CNN networks followed by the FC
    layer (hidden FC + BN + ReLU, then one output per key-point coordinate)."""

    def __init__(self, in_shape: Sequence[int], n_kernels: Sequence[int], kernel_sizes: Sequence[int],
                 pool_sizes: Sequence[int], n_basis_kernels: int, temperature: float, dropout: float,
                 d_hidden: int, n_keypoints: int = 17, keypoint_dims: int = 2):
        super().__init__()
        if not len(n_kernels) == len(kernel_sizes) == len(pool_sizes):
            raise ValueError("n_kernels, kernel_sizes and pool_sizes need one entry per SDy-CNN stack")
        self.n_keypoints, self.keypoint_dims = n_keypoints, keypoint_dims
        chans = [in_shape[0], *n_kernels]
        self.features = nn.Sequential(*[
            SDyCNNBlock(chans[i], chans[i + 1], kernel_sizes[i], pool_sizes[i], n_basis_kernels,
                        temperature, dropout) for i in range(len(n_kernels))])
        with torch.no_grad():
            n_flat = self.features.eval()(torch.zeros(2, *in_shape)).flatten(1).shape[1]
        self.features.train()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, d_hidden), nn.BatchNorm1d(d_hidden),
                                  nn.ReLU(), nn.Linear(d_hidden, n_keypoints * keypoint_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).view(-1, self.n_keypoints, self.keypoint_dims)


class PoseEstimator(nn.Module):
    """Full architecture of Fig. 4: frozen Denoiser (pre-trained AE encoder) + Estimator.
    With ``denoiser=None`` the Estimator works on the CSI directly (clean-data setting)."""

    def __init__(self, estimator: Estimator, denoiser: nn.Module | None = None):
        super().__init__()
        self.denoiser = denoiser
        self.estimator = estimator
        if denoiser is not None:
            for p in denoiser.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.denoiser is not None:
            self.denoiser.eval()    # frozen: BN keeps the statistics learned in stage 1
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.denoiser is not None:
            with torch.no_grad():
                x = self.denoiser(x)
        return self.estimator(x)


def build_autoencoder(cfg: dict) -> DenoisingAutoencoder:
    d = cfg["denoiser"]
    return DenoisingAutoencoder((3, *cfg["data"].get("resize", (136, 32))), d["n_kernels"],
                                d["kernel_sizes"], d["pool_sizes"])


def build_estimator(cfg: dict, in_shape: Tuple[int, ...]) -> Estimator:
    e = cfg["estimator"]
    return Estimator(in_shape, e["n_kernels"], e["kernel_sizes"], e["pool_sizes"], e["n_basis_kernels"],
                     e["temperature"], e["dropout"], e["d_hidden"],
                     keypoint_dims=cfg["data"].get("keypoint_dims", 2))
