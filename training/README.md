# Training of the SR models (reference)

The EDiffSR models were trained with the code in `ediffsr/` (see `ediffsr/README.md`). The ESRGAN models were trained with the code of the Satlas super-resolution project, which is not duplicated here;
`configs_ref/` contains the configuration files of these runs (absolute paths were replaced by `<HOME>`).

| Model | Original code | Config of the run | Checkpoint used in the paper (`WEIGHTS_DIR` name) |
|---|---|---|---|
| EDiffSR, 8 bands, joint (&times;2 / &times;4) | `ediffsr/` of this repository (EDiffSR with the changes for multispectral input) | `ediffsr/codes/config/sisr/options/train/setting2x_sen2venus_l1.yml` | iteration 500 000 &rarr; `ediffsr_8ch.pth` |
| EDiffSR, RGBNir (&times;2) | the same | `.../options/train/setting2xRGB_NIR_val_sen2venus.yml` | iteration 540 000 &rarr; `ediffsr_rgbnir.pth` |
| EDiffSR, IR (&times;4) | the same | `.../options/train/setting4xIR_sen2venus.yml` | iteration 480 000 &rarr; `ediffsr_ir.pth` |
| ESRGAN, 8 bands | [Satlas super-resolution](https://github.com/allenai/satlas-super-resolution) (`SSRESRGANModel`) | `sen2venus_cliploss_esrgan.yml` | iteration 375 000 &rarr; `esrgan_8ch.pth` |
| ESRGAN, RGBNir | the same | `sen2venus_cliploss_esrganRGBNIR.yml` | iteration 385 000 &rarr; `esrgan_rgbnir.pth` |
| ESRGAN, IR | the same | `sen2venus_cliploss_esrgan4IR.yml` | iteration 615 000 &rarr; `esrgan_ir.pth` |

The ESRGAN checkpoints store the weights of the exponential moving average under the key `params_ema`. The EDiffSR checkpoints are plain state dicts of the denoising network (a `module.` prefix is removed by the loader).

Notes on the ESRGAN configs: they are working copies of the runs and contain commented-out alternatives. The training data of both models: reflectance / 10000 clipped to [0, 0.5]; the four 20 m bands resized to 128 &times; 128 and concatenated in front of the four 10 m bands.
