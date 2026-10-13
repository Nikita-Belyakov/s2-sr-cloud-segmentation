from __future__ import annotations

import os
from typing import Literal, Tuple

import numpy as np
import torch
try:  # rasterio is only needed for GeoTIFF tiling, not for patch-wise inference
    import rasterio
    from rasterio.windows import Window
except ImportError:  # pragma: no cover
    rasterio = None
    Window = None
from tqdm import tqdm

from .inference_utils import (
    pad_reflect, 
    split_image_into_overlapping_patches,
    stich_together, 
    unpad_image,
    adjust_histogram,
    get_quantile_ranges,
    get_dtype_from_range
)

class BaseInferenceModel:
    def __init__(
        self, device : str, scale: int = 4, model_path: str | None = None
    ) -> None:
        """ Initialize the model.

        Args:
            device (str): The device to run the model on, e.g. 'cuda' or 'cpu'.
            scale (int, optional): The scale factor of the model. Defaults to 4.
            model_path (str | None, optional): The path to the model weights file. Defaults to None.
        """

        self.device = device
        self.scale = scale
        
    def load_weights(self, model_path: os.PathLike) -> None:
        pass
    
    def predict_batch(self, lr_batch: torch.Tensor) -> torch.Tensor:
        pass
    
    def predict(
        self, 
        lr_image : np.ndarray,
        img_range: Tuple[int, int] = (-0.5, 0.5),
        batch_size:int=4, 
        patches_size:int=224,
        padding:int=16, 
        pad_size:int=16
    ) -> np.ndarray:
        """ Predict high resolution image from low resolution image

        Args:
            `lr_image` (np.ndarray): low resolution image
            `batch_size` (int, optional): batch size. Defaults to 4.
            `patches_size` (int, optional): size of the patches. Defaults to 224.
            `padding` (int, optional): size of the padding of the patches. Defaults to 16.
            `pad_size` (int, optional): size of the padding of the whole image. Defaults to 16.

        Returns:
            np.ndarray: high resolution image
        """
        # Unpack arguments
        scale = self.scale
        device = self.device
        
        # Pad the low resolution image
        lr_image = np.array(lr_image)#[:, :, :3]
        if pad_size > 0:
            lr_image = pad_reflect(lr_image, pad_size)

        # Split the low resolution image into overlapping patches
        patches, p_shape = split_image_into_overlapping_patches(
            lr_image, patch_size=patches_size, padding_size=padding
        )
#         print(p_shape)
        # Convert the patches to tensors and move them to the device
        img = torch.FloatTensor(
            (patches - img_range[0]) / (img_range[1] - img_range[0])
        ).permute((0, 3, 1, 2)).to(device).detach()
#         print(img.shape)
        # Predict the high resolution patches
        res = self.predict_batch(img[0:batch_size])
        for i in range(batch_size, img.shape[0], batch_size):
            res = torch.cat((res, self.predict_batch(img[i:i+batch_size])), 0)

        # Move the high resolution patches to the CPU and convert them to numpy arrays
        sr_image = res.permute((0, 2, 3, 1)).clamp_(-0.5, 0.5).cpu()
        np_sr_image = sr_image.numpy()

        # Scale the padded size and the target shape
        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (8,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (8,)
        
#         print(padded_size_scaled)
#         print(scaled_image_shape)
        # Stitch the high resolution patches together
        if padding > 0:
            np_sr_image = stich_together(
                np_sr_image, padded_image_shape=padded_size_scaled, 
                target_shape=scaled_image_shape, padding_size=padding * scale
            )
#         print(np_sr_image.shape)
        # Scale the image and remove the padding
        sr_img = (np_sr_image * (img_range[1] - img_range[0]) + img_range[0])
#         sr_img = sr_img.astype(get_dtype_from_range(img_range))
        if pad_size > 0:
            sr_img = unpad_image(sr_img, pad_size*scale)

        return sr_img
    
    def predict_geotiff(
        self,
        src: os.PathLike | rasterio.io.DatasetReader,
        save_path: os.PathLike,
        rgb_bands: Tuple[int] = (1, 2, 3),
        in_ranges: Literal['auto'] | Tuple[Tuple[int, int]] | None = None,
        out_range: Tuple[int, int] = (0, 255),
        window_size: int | None = None,
        window_overlap: int | None = None,
        batch_size: int = 4,
        patches_size: int = 224,
        padding: int = 16,
        pad_size: int = 16,
    ) -> None:
        """Predict high resolution image from low resolution geotiff image

        Args:
            `src` (os.PathLike | rasterio.io.DatasetReader): 
                low resolution geotiff source
                if src is a path, it will be opened with rasterio
            `save_path` (os.PathLike):
                path to save the geotiff file
            `rgb_bands` (Tuple[int], optional): 
                bands to use for prediction. Since model is trained on RGB, 
                it is necessary to choose only RGB bands. Defaults to (1, 2, 3).
            `in_ranges` (Literal['auto'] | Tuple[Tuple[int, int]], optional):
                input image range for each channel. 
                if 'auto', it will be computed from the image (0.001 and 0.999 quantiles).
                Defaults to None.
            `out_range` (Tuple[int, int], optional): 
                image range. Defaults to (0, 255).
            `window_size` (int | None, optional): window size.
                if None, try to process the whole image at once
                Recommended to use a multiple of patch_size and subtract 2*pad_size from it.
                Defaults to None.
            `window_overlap` (int | None, optional): 
                window overlap. If None and window_size is not None, it will be set to 2*pad_size. 
                Defaults to None.
            `batch_size` (int, optional): 
                batch size. Defaults to 4.
            `patches_size` (int, optional): 
                size of the patches. Defaults to 224. Real size is `patches_size + 2*padding`.
            `padding` (int, optional): 
                size of the padding of the patches. Defaults to 16.
            `pad_size` (int, optional): 
                size of the reflect padding of the whole image 
                (or window if window_size is not None). Defaults to 16.
        """

        # Check if src is a path or a rasterio dataset
        srс_is_path = isinstance(src, str)
        if srс_is_path:
            src = rasterio.open(src)

        new_height = src.height * self.scale
        new_width = src.width * self.scale
        
        # set window_size and window_overlap if they are None
        if window_overlap is None:
            window_overlap = 2 * pad_size if window_size is not None else 0
        if window_size is None:
            window_size = max(src.height, src.width)
        
        # Update the transform to match the new image size
        transform = src.transform * src.transform.scale(1 / self.scale)

        # Define the meta data for the new geotiff
        meta = {
            "driver": "GTiff",
            "crs": src.crs,
            "height": new_height,
            "width": new_width,
            "count": 3,
            "transform": transform,
            "dtype": get_dtype_from_range(out_range),
        }
        
        # process image in windows and save the new geotiff image
        with rasterio.open(save_path, "w", **meta) as dst:
            # set the progress bar
            total = int(
                np.ceil((src.height-window_overlap)/(window_size-window_overlap))*
                np.ceil((src.width-window_overlap)/(window_size-window_overlap)) 
            ) 
            pbar = tqdm(total = total, desc="Processing")
            
            # Get the input ranges (0.001 and 0.999 quantiles)
            if in_ranges == 'auto':
                input_ranges = get_quantile_ranges(
                    src.read(rgb_bands), channel_last=False
                )
            elif isinstance(in_ranges, (tuple, list, np.ndarray)):
                input_ranges = in_ranges
            
            # Process the image in windows
            x_offsets = range(
                window_overlap//2, 
                src.width-window_overlap//2, 
                window_size-window_overlap
            )
            y_offsets = range(
                window_overlap//2, 
                src.height-window_overlap//2, 
                window_size-window_overlap
            )
            for y_off in y_offsets:
                row_off = y_off - window_overlap // 2
                
                for x_off in x_offsets:
                    col_off = x_off - window_overlap // 2
                    
                    # Get the window
                    window = Window(
                        col_off, row_off, window_size, window_size
                    ).intersection(Window(0, 0, src.width, src.height))
                    
                    # Read the window
                    data = src.read(rgb_bands, window=window)
                    if in_ranges is not None:
                        data = adjust_histogram(
                            data, in_ranges=input_ranges, out_range=out_range
                        )
                    data = data.transpose(1, 2, 0)
                    
                    # Predict the high resolution image
                    sr_data = self.predict(
                        data,
                        img_range=out_range,
                        batch_size=batch_size,
                        patches_size=patches_size,
                        padding=padding,
                        pad_size=pad_size,
                    ).transpose(2, 0, 1)
                    
                    # Drop the overlapping part of the high resolution image 
                    scaled_half_overlap = window_overlap * self.scale // 2
                    if col_off != 0:
                        sr_data = sr_data[:, :, scaled_half_overlap:]
                    if row_off != 0:
                        sr_data = sr_data[:, scaled_half_overlap:, :]

                    # Get the scaled window
                    new_window = Window(
                        (x_off * self.scale) * (col_off != 0), 
                        (y_off * self.scale) * (row_off != 0),
                        window_size * self.scale - scaled_half_overlap * (col_off != 0) , 
                        window_size * self.scale - scaled_half_overlap * (row_off != 0),
                    ).intersection(Window(0, 0, new_width, new_height))

                    # Write the window
                    dst.write(sr_data, window=new_window)

                    pbar.update(1)

        if srс_is_path:
            src.close()


    # DEPRECATED
    def predict_geotiff_v0(self, 
        src: os.PathLike | rasterio.io.DatasetReader | rasterio.io.DatasetWriter,
        return_type: Literal['array', 'geotiff'] | None = 'array',
        save_path: os.PathLike | None = None,
        batch_size: int = 4, patches_size: int = 192,
        padding: int = 24, pad_size: int = 16
    ) -> np.ndarray | rasterio.io.DatasetWriter | None:
        """ Predict high resolution image from low resolution geotiff image
        ! This function is deprecated
        Неудачная реализация, решил, что нет смысла добавлять возможность возвращать массив в памяти,
        по смыслу функции логично было оставить только функционал записи в файл
        кроме того здесь 
        - не был реализован проход окнами по большому файлу
        - не было реализовано приведение гистограммы
        
        Args:
            `src` (os.PathLike | rasterio.io.DatasetReader): low resolution geotiff source
                if src is a path, it will be opened with rasterio
            `return_type` (Literal['array', 'geotiff'], optional): return type. Defaults to 'array'.
                if 'array', return a numpy array
                if 'geotiff', return a rasterio.io.DatasetWriter
                else return None
            `save_path` (os.PathLike, optional): save path. Defaults to None.
                if passed the geotiff will be saved to save_path
            `batch_size` (int, optional): batch size. Defaults to 4.
            `patches_size` (int, optional): size of the patches. Defaults to 192.
            `padding` (int, optional): size of the padding of the patches. Defaults to 24.
            `pad_size` (int, optional): size of the padding of the whole image. Defaults to 16.

        Returns:
            np.ndarray | rasterio.io.DatasetWriter | None: 
                high resolution image or rasterio DatasetWriter or None
        """
        
        # Check if src is a path or a rasterio dataset
        srс_is_path = isinstance(src, str)
        if srс_is_path:
            src = rasterio.open(src)
        
        # Predict high resolution image from low resolution image
        sr_img = self.predict(
            src.read().transpose(1, 2, 0),
            batch_size=batch_size, 
            patches_size=patches_size,
            padding=padding, 
            pad_size=pad_size
        )

        # Update the transform to match the new image size
        transform = src.transform * src.transform.scale(
            (src.width / sr_img.shape[1]), (src.height / sr_img.shape[0])
        )

        meta = {
            'driver'    : 'GTiff',
            'crs'       : src.crs,  
            'count'     : sr_img.shape[2], 
            'width'     : sr_img.shape[1],
            'height'    : sr_img.shape[0], 
            'transform' : transform,
            'dtype'     : sr_img.dtype
        }

        if srс_is_path:
            src.close()
        
        # Save the new geotiff image
        if save_path:
            with rasterio.open(save_path, 'w', **meta) as dst:
                dst.write(sr_img.transpose(2, 0, 1))

        # Return the high resolution image
        if return_type == 'geotiff':
            out = rasterio.io.MemoryFile().open(**meta)
            out.write(sr_img.transpose(2, 0, 1))
            return out
        elif return_type == 'array':
            return sr_img
    