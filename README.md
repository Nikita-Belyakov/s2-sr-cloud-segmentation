# Super-resolution of Sentinel-2 imagery with diffusion and GAN models and its impact on cloud and cloud-shadow segmentation

Code for the evaluation of two super-resolution (SR) models of eight-band Sentinel-2 imagery, the diffusion model **EDiffSR** and the GAN **ESRGAN** (both trained on Sen2VEN&mu;S, 10 m &rarr; 5 m),
and for the downstream experiment: cloud and cloud-shadow segmentation (DeepLabV3+ / EfficientNet-B0) on cloudy Sentinel-2 L1C patches with the original patches, bicubic interpolation, the SR results and 16-channel concatenations as inputs.

## Contents

| Folder | Content |
|---|---|
| `src/sr_benchmark` | evaluation of EDiffSR, ESRGAN and bicubic interpolation on the same Sen2VEN&mu;S validation patches (pixel-wise, spectral, perceptual metrics, ERQA), qualitative figures |
| `src/cloudy_sr` | SR of the cloudy patches, edge (ERQA) and spectral analysis of the SR results |
| `src/segmentation` | training and evaluation of the segmentation networks (fixed split and scene-level cross-validation), boundary and object-size metrics, reports, LaTeX tables |
| `ediffsr/` | EDiffSR code with the changes for multispectral Sentinel-2 / VEN&mu;S data (training script, Sen2VEN&mu;S dataset class, configs); see [ediffsr/README.md](ediffsr/README.md) |
| `s2sr/` | paths, the compact EDiffSR inference module (`s2sr/ediffsr`) and the ESRGAN generator |
| `slurm/`, `configs/` | SLURM templates and the environment template |

## Installation

```bash
git clone https://github.com/Nikita-Belyakov/s2-sr-cloud-segmentation.git
cd s2-sr-cloud-segmentation
python -m venv .venv && source .venv/bin/activate     # Python 3.10
pip install -r requirements.txt                       # install PyTorch for your CUDA first (see requirements.txt)
cp configs/env.example.sh configs/env.sh              # set the paths of the data and of the weights
source configs/env.sh
```

A GPU is required for the SR inference (EDiffSR makes 100 reverse-SDE steps per patch) and for the training of the segmentation networks.

## Data and model weights

The Sen2VEN&mu;S dataset is public ([doi:10.5281/zenodo.6514159](https://doi.org/10.5281/zenodo.6514159)). **The cloudy Sentinel-2 patches and the weights of the SR models are available on request from the author of the paper** (Nikita Belyakov, Skoltech, Nikita.Belyakov@skoltech.ru).

Expected layout (see `s2sr/paths.py`; every path can be set with an environment variable):

```
$S2V_ROOT/npz/<patch>.npz                        # keys: sentinel_10m (4x128x128), sentinel_20m (4x64x64), venus_5m_10m, venus_5m_20m (4x256x256), digital numbers
$S2V_ROOT/splits/{train,valid,test_launch}.csv   # column "filename"
$WORK_DIR/cloudy_split/{train,val,test}/         # 14-channel uint16 tif + png mask (0 clear, 1 cloud, 2 shadow) of the cloudy patches
$WEIGHTS_DIR/{ediffsr,esrgan}_{8ch,rgbnir,ir}.pth
```

## Usage

Run from the repository root; `slurm/*.sbatch` contain the complete command lines.

```bash
# SR benchmark on Sen2VENuS: EDiffSR, ESRGAN and bicubic on the same validation patches
sbatch slurm/sr_eval_sen2venus.sbatch 1024 main

# SR of the cloudy patches by both models
python src/cloudy_sr/split_scenes.py /path/to/unpacked_patches $WORK_DIR/cloudy_split
sbatch slurm/cloudy_sr.sbatch

# segmentation: fixed split and scene-level cross-validation, then reports and tables
sbatch slurm/seg_fixed.sbatch
sbatch slurm/seg_cv.sbatch
sbatch slurm/seg_eval_reports.sbatch

# training of EDiffSR (optional)
sbatch slurm/train_ediffsr.sbatch setting2x_sen2venus_l1.yml
```

Inputs of the segmentation network (`--setup` of `src/segmentation/seg_experiment.py`): `orig` (original patch, 256 &times; 256), `bicubic` (bicubic &times;2, 512 &times; 512), `ediffsr` / `esrgan` (SR result),
`orig_ediffsr` / `orig_esrgan` (16 channels: bicubic &times;2 of the original + SR result), `resid_ediffsr` (bicubic &times;2 + (EDiffSR &minus; bicubic), fixed split only).
All inputs are scored on the common 256 &times; 256 grid.

## Notes

* All 8-band tensors use the Sen2VEN&mu;S order `[B5, B6, B7, B8A, B2, B3, B4, B8]`; reflectance = DN / 10000 clipped to `[0, 0.5]`.
* EDiffSR sampling is stochastic (the scripts fix the seed; GPU non-determinism remains).
* The ESRGAN generator is a &times;4 network; its output is averaged over 2 &times; 2 blocks to the 5 m grid, so that both models give 8 bands on the same grid. The checkpoints expect the input divided by 255.
* LPIPS and FID download the AlexNet / Inception weights on the first run.

Third-party code and data licenses are listed in [NOTICE.md](NOTICE.md).
