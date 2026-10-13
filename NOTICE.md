# Third-party code and data

| Path | Origin | License |
|---|---|---|
| `ediffsr/`, `s2sr/ediffsr/` | **EDiffSR**: Y. Xiao et al., *EDiffSR: An Efficient Diffusion Probabilistic Model for Remote Sensing Image Super-Resolution*, IEEE TGRS, 2024, https://github.com/XY-boy/EDiffSR , with changes for multispectral input (listed in `ediffsr/README.md`). The denoising network and the SDE utilities derive from **IR-SDE** (Z. Luo et al., ICML 2023, https://github.com/Algolzw/image-restoration-sde) and **NAFNet** (L. Chen et al., ECCV 2022). | IR-SDE: MIT. The EDiffSR repository does not state a license (checked when this repository was prepared); the files are published with attribution to the original authors, and the original terms apply to the unmodified parts. |
| `s2sr/esrgan_net.py` | RRDBNet generator adapted from **BasicSR** (https://github.com/XPixelGroup/BasicSR) and the **Satlas super-resolution** code (https://github.com/allenai/satlas-super-resolution), the architecture used to train the ESRGAN checkpoints. | Apache-2.0 |
| `training/configs_ref/` | Reference copies of the configuration files of the ESRGAN training runs (Satlas super-resolution configs). | as the origin repository |

Data: Sen2VEN&mu;S (J. Michel, J. Vinasco-Salinas, J. Inglada, O. Hagolle, 2022, https://doi.org/10.5281/zenodo.6514159); the Sentinel-2 files are under the Etalab Open Licence 2.0, the VEN&mu;S files under CC BY-NC 4.0 (non-commercial use only).
