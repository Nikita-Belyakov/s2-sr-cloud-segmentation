# Super-resolution of Sentinel-2 imagery with diffusion and GAN models and its impact on cloud and cloud-shadow segmentation

Code of the paper

> N. Belyakov. *Multispectral Super-Resolution of Sentinel-2 Imagery with Diffusion and GAN Models: Evaluation on Sen2VEN&mu;S and Impact on Cloud and Cloud-Shadow Segmentation.* (submitted to *Advances in Space Research*)  <!-- TODO: title / status / DOI -->

The repository contains the evaluation of two super-resolution (SR) models of eight-band Sentinel-2 imagery (the diffusion model **EDiffSR** and the GAN **ESRGAN**, both trained on Sen2VEN&mu;S, 10 m &rarr; 5 m)
and the downstream experiment: cloud and cloud-shadow segmentation (DeepLabV3+ / EfficientNet-B0) on 5342 cloudy Sentinel-2 L1C patches with the original patches, bicubic interpolation, the SR results and 16-channel concatenations as inputs.

| Part | What it does | Paper |
|---|---|---|
| `src/sr_benchmark` | one script that scores EDiffSR, ESRGAN and bicubic interpolation on the **same** Sen2VEN&mu;S validation patches; ERQA against the VEN&mu;S reference; qualitative figures | Tables 1&ndash;3, Figures 2&ndash;4 |
| `src/cloudy_sr` | SR of the cloudy patches, ERQA against the interpolated source, spectral analysis of the SR results | Table 4, Figures 5&ndash;6 |
| `src/segmentation` | training / evaluation of the segmentation networks (fixed split and scene-level cross-validation), boundary and object-size metrics, reports, LaTeX tables | Tables 5&ndash;10, Figure 7, Appendix |
| `src/data_checks` | statistics of the cloudy patches that justify the pre-processing (B10-like channel, cloud flag, brightness of thin clouds) | Section 3 |
| `src/graphical_abstract` | thumbnails and an editable PPTX of the graphical abstract | &mdash; |
| `s2sr/` | paths (`paths.py`), the inference wrapper of EDiffSR (`ediffsr/`) and the ESRGAN generator (`esrgan_net.py`, loads the checkpoints without BasicSR / Satlas) | &mdash; |
| `ediffsr/` | **EDiffSR code with the changes for multispectral Sentinel-2 / VEN&mu;S data**: training script, Sen2VEN&mu;S dataset class, configs of the three models; see [ediffsr/README.md](ediffsr/README.md) | training of the SR models |
| `slurm/`, `configs/` | SLURM templates and the environment template | &mdash; |
| `training/` | which config / checkpoint of the paper belongs to which model; reference configs of the ESRGAN runs | &mdash; |

## Installation

```bash
git clone https://github.com/Nikita-Belyakov/s2-sr-cloud-segmentation.git   # TODO: check the final URL
cd s2-sr-cloud-segmentation
python -m venv .venv && source .venv/bin/activate          # Python 3.10
pip install -r requirements.txt                            # install PyTorch for your CUDA first, see the comment in requirements.txt
cp configs/env.example.sh configs/env.sh                   # then edit the paths
source configs/env.sh
```

A GPU is needed for the SR inference (EDiffSR makes 100 reverse-SDE steps per patch) and for the training of the segmentation networks (about 2.5 h per cross-validation fold and seed on one A100).
The metrics (LPIPS, FID) download the weights of AlexNet and Inception on the first use; on a cluster without internet access copy the cache of `torch.hub` to the compute node.

## Data and weights

The data and the checkpoints are not stored in git (see `.gitignore`). Set the locations with the variables of `configs/env.example.sh`.

**Sen2VEN&mu;S** (Michel et al., 2022; [10.5281/zenodo.6514159](https://doi.org/10.5281/zenodo.6514159)). The scripts read it as one compressed `npz` file per patch (`S2V_ROOT/npz/<name>.npz`) with the arrays

| key | shape | content (digital numbers, reflectance &times; 10<sup>4</sup>) |
|---|---|---|
| `sentinel_10m` | 4 &times; 128 &times; 128 | Sentinel-2 B2, B3, B4, B8 (10 m) |
| `sentinel_20m` | 4 &times; 64 &times; 64 | Sentinel-2 B5, B6, B7, B8A (20 m); resized to 128 &times; 128 by the scripts |
| `venus_5m_10m` | 4 &times; 256 &times; 256 | VEN&mu;S bands that correspond to `sentinel_10m` (5 m) |
| `venus_5m_20m` | 4 &times; 256 &times; 256 | VEN&mu;S bands that correspond to `sentinel_20m` (5 m) |

and the split files `S2V_ROOT/splits/{train,valid,test_launch}.csv` with the column `filename` (99&nbsp;716 / 33&nbsp;239 / 100 patches). The split is by patch. The conversion from the distribution of the dataset to this layout is not part of the repository. <!-- TODO: add the conversion script or the link to the converted data -->

**Cloudy Sentinel-2 patches.** 5342 patches of 256 &times; 256 pixels (10 m) of 8 scenes: 14-channel `uint16` GeoTIFFs (12 spectral bands, a low-amplitude channel resembling B10 and a binary flag) and `png` masks with the values 0 clear, 1 cloud, 2 shadow, named `mask84_<scene>_<row>_<col>`.
<!-- TODO: state where the data are available (the archive patched_Sentinel_cloudy_scenes.zip) -->
`src/cloudy_sr/split_scenes.py <unpacked patches> $WORK_DIR/cloudy_split` creates `cloudy_split/{train,val,test}` and `split_manifest.csv` (the fixed scene-level split; the folds of the cross-validation are defined in `src/segmentation/seg_experiment.py`).

**Checkpoints** of the six SR models (put them into `WEIGHTS_DIR` under these names or set the variables of `s2sr/paths.py`):

| file | model | input &rarr; output |
|---|---|---|
| `ediffsr_8ch.pth` | EDiffSR, joint | 8 bands (10 m &times;2, 20 m &times;4) &rarr; 8 bands at 5 m |
| `ediffsr_rgbnir.pth` | EDiffSR | B2, B3, B4, B8 (&times;2) |
| `ediffsr_ir.pth` | EDiffSR | B5, B6, B7, B8A (&times;4) |
| `esrgan_8ch.pth` | ESRGAN, joint | 8 bands &rarr; 8 bands at 5 m |
| `esrgan_rgbnir.pth` | ESRGAN | B2, B3, B4, B8 |
| `esrgan_ir.pth` | ESRGAN | B5, B6, B7, B8A |

<!-- TODO: link to the published checkpoints (Zenodo / Hugging Face) -->

## Reproducing the paper

All scripts take their locations from `s2sr/paths.py` (`WORK_DIR` is the folder where the outputs are written and where `cloudy_split*` live). The SLURM templates in `slurm/` contain the exact command lines; they are run from the repository root.

```bash
# 1. SR benchmark on Sen2VENuS (Tables 1-2: metrics, bands; Figures 2-3)
sbatch slurm/sr_eval_sen2venus.sbatch 1024 main          # 1024 validation patches, 3 stages + tables
sbatch slurm/scene_figs.sbatch
sbatch slurm/erqa_sen2venus.sbatch                       # Table 3, Figure 4

# 2. SR of the cloudy patches and its analysis (Table 4, Figures 5-6)
python src/cloudy_sr/split_scenes.py /path/to/unpacked_patches $WORK_DIR/cloudy_split
sbatch slurm/cloudy_sr.sbatch                            # --array=0-11, both models
sbatch slurm/erqa_cloudy.sbatch

# 3. segmentation (Tables 5-10, Figure 7, Appendix)
sbatch slurm/seg_fixed.sbatch                            # 7 inputs x 3 seeds, fixed split
sbatch slurm/seg_cv.sbatch                               # 4 folds x 2 seeds x 6 inputs
sbatch slurm/seg_eval_reports.sbatch                     # evaluation, reports, LaTeX tables
```

The tables are written as `.tex` files to `$WORK_DIR/tables`; the reports (`seg_report/`, `seg_cv_report/`, `seg_small_report/`, `seg_thin_report/`) are markdown files with all numbers.

Inputs of the segmentation (the `--setup` argument of `seg_experiment.py`):

| `--setup` | input of the network |
|---|---|
| `orig` | the original patch, 8 channels, 256 &times; 256 |
| `bicubic` | the original patch interpolated bicubically (&times;2) to 512 &times; 512, 8 channels |
| `ediffsr`, `esrgan` | the SR result, 8 channels, 512 &times; 512 |
| `orig_ediffsr`, `orig_esrgan` | 16 channels: bicubic &times;2 of the original + the SR result |
| `resid_ediffsr` | 16 channels: bicubic &times;2 of the original + (EDiffSR &minus; bicubic) (fixed split only) |

All inputs are scored on the common 256 &times; 256 grid (probabilities of the 512 &times; 512 inputs are averaged over 2 &times; 2 blocks).

## Implementation notes

* **Channel order.** All 8-channel tensors use the order of the Sen2VEN&mu;S data `[B5, B6, B7, B8A, B2, B3, B4, B8]`; in the 14-channel cloudy patches these are the channel indices `[4, 5, 6, 8, 1, 2, 3, 7]`. Reflectance is `DN / 10000` clipped to `[0, 0.5]` as in the training data of the SR models.
* **EDiffSR** receives the bicubically upsampled input (&times;2) scaled to `[-1, 1]` and returns the result in the same range; 100 reverse-SDE steps (IR-SDE, NAFNet denoiser). The sampling is stochastic; the scripts set the seed, but GPU non-determinism remains.
* **ESRGAN.** The generator (`s2sr/esrgan_net.py`) has a fixed &times;4 up-sampling. To obtain the 5 m grid its output is averaged over 2 &times; 2 blocks, so that both models return 8 bands on the same grid. The checkpoints expect the input divided by 255 as in training (`net(x / 255)`); the rows `esrgan_testscript` in the output of `sr_eval_unified.py` show the effect of the other scaling that is used in the separate test script of the training code.
* **Metrics** of the SR benchmark (`sr_eval_unified.py`): PSNR (data range 1), SSIM (Gaussian window 11, `torchmetrics`), RMSE, SAM (degrees), ERGAS (ratio 2 for the 10 m bands and 4 for the 20 m bands), LPIPS (AlexNet) and FID (64-d Inception feature) on the true-colour composite B4, B3, B2; ERQA with the package `erqa` (Canny 100/200, 1 px tolerance). Degenerate patches (fewer than 0.1 % reference edge pixels) are excluded from the ERQA averages.
* **Segmentation.** DeepLabV3+ with a pretrained EfficientNet-B0 encoder (`segmentation_models_pytorch`), AdamW (lr 5e-4, weight decay 1e-2), cosine schedule, 30 epochs, batch 16, cross-entropy + Dice, flips / rot90 augmentations, the epoch with the best validation mean IoU of shadow and cloud is used; per-band standardization with the statistics of the original training patches.

## EDiffSR code and the changes for multispectral imagery

The folder `ediffsr/` publishes the code of [EDiffSR](https://github.com/XY-boy/EDiffSR) (Xiao et al., IEEE TGRS 2024; it builds on [IR-SDE](https://github.com/Algolzw/image-restoration-sde), Luo et al., MIT) **together with our changes** that are needed for eight-band Sentinel-2 / VEN&mu;S data: the number of input / output channels of the denoiser (8 or 4 instead of 3), the Sen2VEN&mu;S dataset class, the training script and the configs of the three models of the paper. The table of all changed files is in [ediffsr/README.md](ediffsr/README.md); `s2sr/ediffsr` is the compact inference version (same network and sampler) that the experiments use.

`s2sr/esrgan_net.py` is adapted from [BasicSR](https://github.com/XPixelGroup/BasicSR) and the [Satlas super-resolution](https://github.com/allenai/satlas-super-resolution) code (Apache-2.0). See [NOTICE](NOTICE.md).
The ERQA metric is the package `erqa` of the MSU Video Group ([Kirillova et al., 2022](https://github.com/msu-video-group/erqa)).

## Citation

```bibtex
@article{belyakov2026sr_cloud,        % TODO: complete after the publication
  author  = {Belyakov, Nikita},
  title   = {Multispectral Super-Resolution of Sentinel-2 Imagery with Diffusion and GAN Models: Evaluation on Sen2VEN$\mu$S and Impact on Cloud and Cloud-Shadow Segmentation},
  journal = {Advances in Space Research},
  year    = {2026}
}
```

## License

<!-- TODO: choose the license of the code (the third-party parts keep their own licenses, see NOTICE.md). -->
