# EDiffSR for multispectral Sentinel-2 imagery

This folder contains the training code of the SR models of the paper: the code of **EDiffSR** (Y. Xiao et al., *EDiffSR: An Efficient Diffusion Probabilistic Model for Remote Sensing Image Super-Resolution*, IEEE TGRS 2024,
<https://github.com/XY-boy/EDiffSR>) with the changes that are needed to train and run it on eight-band Sentinel-2 / VEN&mu;S data. EDiffSR itself is built on IR-SDE (Z. Luo et al., ICML 2023, <https://github.com/Algolzw/image-restoration-sde>, MIT license)
and NAFNet; the original code and its license terms are those of the repositories above.

The folder keeps the layout of the original repository (`codes/config/sisr`, `codes/data`, `codes/utils`) so that the files can be compared with the upstream ones. Only the files needed for training are included.
The inference that is used in the experiments of the paper is implemented in `s2sr/ediffsr` (the same network and sampler; it loads the checkpoints produced here).

## Changes with respect to the original code

| File | Change |
|---|---|
| `codes/config/sisr/train_sen2venus.py` | **new**: training script (based on `train.py` of the original) for Sen2VEN&mu;S: reads the patches and the split csv files (`S2V_NPZ`, `S2V_SPLITS`), builds the training / validation sets with `Sen2VenusDataset`, validates with PSNR / SSIM (`torchmetrics`) and saves RGB / IR preview triplets (Sentinel-2, SR, VEN&mu;S) as png; the preview folders are created on demand and the metrics are placed on GPU 0 when the script is run without the distributed launcher |
| `codes/data/SuperResolution.py` | **new**: `Sen2VenusDataset`. Reads an `npz` patch, reflectance = DN / 10000 clipped to [0, 0.5]; `fuse=True`, `only_IR=False`: the four 20 m bands (B5, B6, B7, B8A) are resized to 128 &times; 128 and concatenated in front of the four 10 m bands (B2, B3, B4, B8), the pair is bicubically upsampled to 256 &times; 256 and the target is the 8-band 5 m VEN&mu;S patch (order B5, B6, B7, B8A, B2, B3, B4, B8); `only_IR=True` gives the four 20 m bands only (&times;4 model), `fuse=False` the four 10 m bands only (RGBNir, &times;2 model) |
| `codes/config/sisr/models/modules/DenoisingNAFNet_arch.py` | default arguments of `ConditionalNAFNet` (`img_channel=8`, `width=256`) and of `NAFBlock` (`c=64`); the number of input / output channels is set in the config (`network_G.setting.img_channel`: 8 for the joint model, 4 for the RGBNir and IR models) instead of 3 |
| `codes/config/sisr/models/denoising_model.py` | weight decay of the generator fixed to `1e-5`; `get_current_visuals` takes the index of the sample of the batch |
| `codes/config/sisr/models/__init__.py` | logging of the creation of the model switched off |
| `codes/config/sisr/options.py` | the memcached branch of the dataset options is commented out |
| `codes/data/__init__.py` | `create_dataloader`: the same loader for train and validation (shuffling and the sampler as in training), `spawn` multiprocessing context |
| `codes/data/LQGT_dataset.py` | scale list taken from the options (`scale_list`) and random re-scaling that keeps the aspect ratio (not used for Sen2VEN&mu;S) |
| `codes/utils/img_utils.py` | `tensor2img`: the channels 4:7 of an 8-band tensor (B4, B3, B2 in the Sen2VEN&mu;S order) are used as the RGB preview; arrays are converted with `np.array` before PSNR / SSIM |
| `codes/utils/sde_utils.py` | default number of reverse steps `T=100` in `reverse_sde` / `reverse_ode`; the progress bar of the sampler is switched off |
| `codes/config/sisr/options/train/*.yml` | the three training configs of the paper (below); paths are relative |

All other files (`base_model.py`, `lr_scheduler.py`, `networks.py`, `optimizer.py`, `loss.py`, `module_util.py`, the other dataset classes ...) are unchanged.

## Training

```bash
cd ediffsr/codes/config/sisr
export S2V_NPZ=/path/to/sen2venus/npz S2V_SPLITS=/path/to/sen2venus/splits
python -m torch.distributed.launch --nproc_per_node=4 --master_port=13122 --use_env train_sen2venus.py \
       -opt options/train/setting2x_sen2venus_l1.yml --launcher pytorch
```

| Model of the paper | Config |
|---|---|
| 8 bands, joint (&times;2 for 10 m, &times;4 for 20 m bands, L1 loss) | `setting2x_sen2venus_l1.yml` |
| RGBNir, &times;2 | `setting2xRGB_NIR_val_sen2venus.yml` |
| IR, &times;4 | `setting4xIR_sen2venus.yml` |

The configs are the working configs of the runs: `pretrain_model_G` and `resume_state` are set to `~` (training from scratch); the original runs were resumed several times from the saved states.
The generator is a conditional NAFNet (width 64, encoder blocks `[14, 1, 1, 1]`, one middle block, decoder blocks `[1, 1, 1, 1]`) trained as the denoiser of the IR-SDE (`max_sigma=50`, `T=100`, cosine schedule, `eps=0.005`) with the AdamW optimizer, L1 loss, `lr_G = 4e-5` with cosine annealing, 500 k iterations, batch 12 (see the yml files).
A checkpoint `<iter>_G.pth` of `experiments/sisr/<name>/models/` is what `WEIGHTS_DIR/ediffsr_*.pth` of the main repository expects (the iterations used in the paper are given in `training/README.md`).
