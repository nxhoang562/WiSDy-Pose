# WiSDy-Pose: Robust WiFi Sensing-Based Human Pose Estimation with a Denoising Autoencoder and Subcarrier-Focusing Dynamic CNN

**WiSDy-Pose** (WiFi + SDy-CNN + Pose) is the official implementation of

> Xuan Hoang Nguyen, Van-Dinh Nguyen, Quang-Trung Luu, Toan Dinh Gian, Oh-Soon Shin,
> **"Robust WiFi Sensing-Based Human Pose Estimation Using Denoising Autoencoder and CNN With Dynamic Subcarrier Attention,"**
> *IEEE Internet of Things Journal*, vol. 12, no. 11, pp. 17066–17079, 2025.

**Paper:** [IEEE Xplore (DOI 10.1109/JIOT.2025.3535156)](https://doi.org/10.1109/JIOT.2025.3535156) ·
[HAL open access (hal-05516654)](https://hal.science/hal-05516654) ·
[PDF in this repository](paper/hoang2025hpe.pdf)

## The problem

Human pose estimation (HPE) locates the key-points of a human body, such as elbows, knees
and shoulders. Cameras and wearable sensors are the usual sources, but cameras raise
privacy concerns and depend on lighting, and wearables are intrusive. **WiFi sensing**
estimates the pose from the channel state information (CSI) of ordinary WiFi devices.
CSI describes how the signal on every OFDM subcarrier is reflected, diffracted and
scattered by the body. WiFi sensing is device-free, cheap, works in the dark and preserves
privacy.

Two difficulties limit WiFi-based pose estimators:

- **Noise.** Real CSI contains strong random noise. Most estimators are trained on heavily
  pre-processed data and degrade in real deployments.
- **Uneven information across subcarriers.** Only some subcarriers carry pose-related
  information. CNNs with fixed kernels treat all of them alike. Self-attention models can
  focus on informative parts, but they suit long sequences, while a CSI sample is a short
  burst of packets, and they are expensive to run.

## The paper

The paper proposes a two-stage architecture trained on CSI amplitudes:

- **Denoiser:** an autoencoder that removes random noise and compresses the CSI.
- **Estimator (SDy-CNN):** a lightweight *subcarrier-focusing dynamic CNN*. Its kernels
  adapt to every input through an attention over subcarriers, a cheap alternative to
  self-attention.
- **Bayesian optimisation:** sets the hyperparameters of both modules, including the
  depth of the Estimator, instead of hand tuning.

On the MM-Fi dataset (40 subjects, 27 activities, three protocols), the architecture
outperforms WiSPPN, Mi-Mose, PerUnet, CSIFormer, WPFormer and CKDformer:

- **Clean data:** up to 8.38% higher PCK@20 than the best previous model.
- **Noisy data (10–30% noise):** up to 14% higher accuracy than the best denoising model,
  PerUnet, while running 12.3 times faster.

## Overview of WiSDy-Pose

WiSDy-Pose estimates 17 human key-points from WiFi CSI. It has two modules (Fig. 4
of the paper):

1. **Denoiser:** a convolutional autoencoder trained to reconstruct clean CSI from noisy
   CSI. After training, only the encoder is kept (transfer learning) and frozen. It removes
   random noise and reduces the dimensionality of the CSI.
2. **Estimator:** a stack of **SDy-CNN** networks (subcarrier-focusing dynamic CNN)
   followed by a fully connected layer. Each SDy-CNN does three things:
   - averages its input over time;
   - computes an attention weight for every subcarrier with two 1-D convolutions;
   - uses these weights to combine *K* basis kernels into an input-dependent kernel.

   As a result, the network focuses on the subcarriers that carry the most pose information.

The hyperparameters of both modules, including the number of SDy-CNN stacks, are selected
by **Bayesian optimisation**.

```
noisy CSI (3 x 136 x 32) ──► Denoiser (frozen AE encoder) ──► code (N_c' x N_sc' x T')
        ──► SDy-CNN x n_stacks ──► FC ──► 17 key-points
```

## Repository structure

```
wisdypose/
  data.py       MM-Fi WiFi-CSI loading, S1/S2/S3 splits, normalisation, noise injection
  models.py     DenoisingAutoencoder, SubcarrierAttention, SDyConv2d, Estimator, PoseEstimator
  engine.py     stage-1 / stage-2 training loops and evaluation
  metrics.py    MPJPE, PA-MPJPE, PCK@alpha
  pipeline.py   end-to-end experiment (data -> stage 1 -> stage 2 -> test)
configs/
  protocol{1,2,3}_original.yaml       clean data, Estimator only        (Table V)
  protocol1_noise10.yaml              10% noise, Denoiser + Estimator   (Tables IV, V)
  protocol2_noise20.yaml              20% noise
  protocol3_noise30.yaml              30% noise
  bo_search_space.yaml                Bayesian-optimisation search space (Table III)
train.py        train and test one configuration
evaluate.py     evaluate a saved model
bo_search.py    hyperparameter search with Bayesian optimisation
tests/          unit tests and an end-to-end run on synthetic MM-Fi-shaped data
paper/          the paper (PDF)
LICENSE         MIT License
```

## Installation

```bash
git clone https://github.com/nxhoang562/WiSDy-Pose.git
cd WiSDy-Pose
pip install -r requirements.txt
```

The code requires Python ≥ 3.9 and PyTorch ≥ 1.13. `wandb` is needed only for `bo_search.py`.

## Data

The experiments use the WiFi modality of the [MM-Fi dataset](https://ntu-aiot-lab.github.io/mm-fi)
(Yang et al., NeurIPS 2023; [toolbox](https://github.com/ybhbingo/MMFi_dataset)). Download
it and keep the original layout:

```
MMFi/
  E01/ S01/ A01/ ground_truth.npy
                 wifi-csi/ frame001.mat ... frame297.mat
  ...
  E04/ S40/ A27/ ...
```

Each frame holds `CSIamp` of shape 3 x 114 x 10 (receive antennas x subcarriers x packets).
Pre-processing follows Section V-B of the paper:

1. **Cleaning:** inf/NaN amplitudes are replaced by the mean of their packet.
2. **Normalisation:** every frame is min-max normalised to [0, 1] (Eq. 12).
3. **Resizing:** frames are resized to 136 x 32 by nearest-neighbour interpolation.
4. **Noise injection:** noisy versions add `noise_level` x N(0, 1) noise, then every
   antenna channel is min-max normalised again. `noise_level` 0.1 / 0.2 / 0.3 corresponds
   to 10% / 20% / 30% noise. The Denoiser is trained to map the noisy frame to the clean
   frame.

Noise is drawn again at every training step. For validation and test it is fixed per sample.

| Option | Values |
|---|---|
| Protocol | `protocol1` (14 daily activities), `protocol2` (13 rehabilitation exercises), `protocol3` (all 27) |
| Split | `random` (S1, default), `cross_subject` (S2), `cross_environment` (S3) |

The held-out part of each split is divided in half into a validation set and a test set.
Model selection uses the validation set only.

## Training

```bash
# 10% noise, Protocol 1: stage 1 (Denoiser) + stage 2 (Estimator) + test
python train.py --config configs/protocol1_noise10.yaml --data-root /path/to/MMFi

# clean data, Estimator only
python train.py --config configs/protocol1_original.yaml --data-root /path/to/MMFi

# cross-subject (S2) or cross-environment (S3) split
python train.py --config configs/protocol3_noise30.yaml --data-root /path/to/MMFi --split cross_environment
```

Each run writes three files to `runs/<config>_<split>/`:
- `autoencoder.pt`
- `model.pt`
- `results.json`: validation and test MPJPE, PA-MPJPE, PCK@20/30/40/50 and inference time per sample.

To evaluate a saved model again:

```bash
python evaluate.py --checkpoint runs/protocol1_noise10_random/model.pt --data-root /path/to/MMFi
```

Training details (Section V-B-3):

| Stage | Optimiser | Batch size | Notes |
|---|---|---:|---|
| Denoiser | RMSProp | 256 | Learning-rate schedule of Eq. (5): ζ₀ for 45 epochs, then ζ₀·e^-0.1 |
| Estimator | Adam | 64 | MSE loss on the key-points |

Both stages run for at most 100 epochs and keep the weights with the lowest validation loss.

## Bayesian optimisation

```bash
wandb login
python bo_search.py --base configs/protocol1_noise10.yaml --data-root /path/to/MMFi --trials 30
```

- **Base config:** fixes the data setting (protocol, noise level, split).
- **Search space:** `configs/bo_search_space.yaml` covers the hyperparameters of Table III,
  including the number of SDy-CNN stacks.
- **Objective:** each trial trains both stages and is scored by the Estimator's validation loss.

The configs in `configs/` contain the optimal values reported in Tables IV and V.

## Results reported in the paper (MM-Fi, S1 random split)

| Setting | MPJPE (mm) ↓ | PA-MPJPE (mm) ↓ | PCK@20 | PCK@30 | PCK@40 | PCK@50 |
|---|---:|---:|---:|---:|---:|---:|
| Protocol 1, original | 133.36 | 93.51 | 60.61 | 74.36 | 82.52 | 87.33 |
| Protocol 1, 10% noise | 133.56 | 93.83 | 59.85 | 74.11 | 82.14 | 87.11 |
| Protocol 2, original | 150.75 | 85.55 | 47.58 | 66.42 | 77.80 | 85.00 |
| Protocol 2, 20% noise | 155.47 | 86.84 | 45.89 | 65.02 | 76.69 | 84.13 |
| Protocol 3, original | 146.47 | 91.88 | 57.48 | 72.72 | 81.34 | 86.53 |
| Protocol 3, 30% noise | 152.22 | 92.43 | 53.51 | 69.65 | 79.09 | 85.32 |

The paper also reports:
- comparisons with WiSPPN, Mi-Mose, PerUnet, CSIFormer, WPFormer and CKDformer (Table VII);
- results for the S2/S3 splits (Table VIII);
- execution times (Fig. 11).

## Tests

```bash
pip install pytest scikit-learn
pytest -q tests
```

The tests check:
- the model shapes for every config;
- that the SDy-CNN equals per-subcarrier kernel aggregation (Eq. 9);
- the metrics;
- an end-to-end run of both training stages on a small synthetic dataset with the MM-Fi layout.

## Citation

If you use WiSDy-Pose or this code, please cite:

```bibtex
@article{nguyen2025robust,
  title   = {Robust WiFi Sensing-Based Human Pose Estimation Using Denoising Autoencoder and CNN With Dynamic Subcarrier Attention},
  author  = {Nguyen, Xuan Hoang and Nguyen, Van-Dinh and Luu, Quang-Trung and Gian, Toan Dinh and Shin, Oh-Soon},
  journal = {IEEE Internet of Things Journal},
  volume  = {12},
  number  = {11},
  pages   = {17066--17079},
  year    = {2025},
  doi     = {10.1109/JIOT.2025.3535156}
}
```

## License

The code is released under the [MIT License](LICENSE). The paper PDF in `paper/` is
provided for reference; its copyright remains with the authors and the publisher.

## Acknowledgements

We thank the authors of the **MM-Fi dataset**, Jianfei Yang, He Huang, Yunjiao Zhou,
Xinyan Chen, Yuecong Xu, Shenghai Yuan, Han Zou, Chris Xiaoxuan Lu and Lihua Xie, for
collecting and publicly releasing the multi-modal data, including the WiFi CSI and
key-point annotations, on which all the experiments of this work are based. Our data
splits follow their [MM-Fi toolbox](https://github.com/ybhbingo/MMFi_dataset). If you use
the data, please also cite their paper:

```bibtex
@inproceedings{yang2023mmfi,
  title     = {{MM-Fi}: Multi-Modal Non-Intrusive {4D} Human Dataset for Versatile Wireless Sensing},
  author    = {Yang, Jianfei and Huang, He and Zhou, Yunjiao and Chen, Xinyan and Xu, Yuecong and Yuan, Shenghai and Zou, Han and Lu, Chris Xiaoxuan and Xie, Lihua},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year      = {2023}
}
```

This work was supported in part by the VinUniversity Seed Grant Program. The SDy-CNN builds
on dynamic convolution (Chen et al., CVPR 2020) and frequency dynamic convolution (Nam et
al., 2022).
