import numpy as np


def pad_reflect(
    image: np.ndarray, 
    pad_size: int
) -> np.ndarray:
    """Pad the image with reflect padding

    Args:
        `image` (np.ndarray): image to be padded
        `pad_size` (int): size of the padding

    Returns:
        np.ndarray: padded image
    """
    imsize = image.shape
    height, width = imsize[:2]
    new_img = np.zeros(
        [height + pad_size * 2, width + pad_size * 2, imsize[2]]
    )#.astype(np.uint16)
    
    new_img[pad_size:-pad_size, pad_size:-pad_size, :] = image

    new_img[0:pad_size, pad_size:-pad_size, :] = np.flip(
        image[0:pad_size, :, :], axis=0
    )  # top
    new_img[-pad_size:, pad_size:-pad_size, :] = np.flip(
        image[-pad_size:, :, :], axis=0
    )  # bottom
    new_img[:, 0:pad_size, :] = np.flip(
        new_img[:, pad_size : pad_size * 2, :], axis=1
    )  # left
    new_img[:, -pad_size:, :] = np.flip(
        new_img[:, -pad_size * 2 : -pad_size, :], axis=1
    )  # right

    return new_img

def unpad_image(image, pad_size):
    return image[pad_size:-pad_size, pad_size:-pad_size, :]


def pad_patch(image_patch, padding_size, channel_last=True):
    """ Pads image_patch with padding_size edge values. """
    
    if channel_last:
        return np.pad(
            image_patch,
            ((padding_size, padding_size), (padding_size, padding_size), (0, 0)),
            'edge',
        )
    else:
        return np.pad(
            image_patch,
            ((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
            'edge',
        )


def unpad_patches(image_patches, padding_size):
    return image_patches[:, padding_size:-padding_size, padding_size:-padding_size, :]


def split_image_into_overlapping_patches(image_array, patch_size, padding_size=2):
    """ Splits the image into partially overlapping patches.
    The patches overlap by padding_size pixels.
    Pads the image twice:
        - first to have a size multiple of the patch size,
        - then to have equal padding at the borders.
    Args:
        image_array: numpy array of the input image.
        patch_size: size of the patches from the original image (without padding).
        padding_size: size of the overlapping area.
    """
    
    xmax, ymax, _ = image_array.shape
    x_remainder = xmax % patch_size
    y_remainder = ymax % patch_size
    
    # modulo here is to avoid extending of patch_size instead of 0
    x_extend = (patch_size - x_remainder) % patch_size
    y_extend = (patch_size - y_remainder) % patch_size
    
    # make sure the image is divisible into regular patches
    extended_image = np.pad(image_array, ((0, x_extend), (0, y_extend), (0, 0)), 'edge')
    
    # add padding around the image to simplify computations
    padded_image = pad_patch(extended_image, padding_size, channel_last=True)
    
    xmax, ymax, _ = padded_image.shape
    patches = []
    
    x_lefts = range(padding_size, xmax - padding_size, patch_size)
    y_tops = range(padding_size, ymax - padding_size, patch_size)
    
    for x in x_lefts:
        for y in y_tops:
            x_left = x - padding_size
            y_top = y - padding_size
            x_right = x + patch_size + padding_size
            y_bottom = y + patch_size + padding_size
            patch = padded_image[x_left:x_right, y_top:y_bottom, :]
            patches.append(patch)
    
    return np.array(patches), padded_image.shape


def stich_together(patches, padded_image_shape, target_shape, padding_size=4):
    """ Reconstruct the image from overlapping patches.
    After scaling, shapes and padding should be scaled too.
    Args:
        patches: patches obtained with split_image_into_overlapping_patches
        padded_image_shape: shape of the padded image contructed in split_image_into_overlapping_patches
        target_shape: shape of the final image
        padding_size: size of the overlapping area.
    """
    
    xmax, ymax, _ = padded_image_shape
    patches = unpad_patches(patches, padding_size)
    patch_size = patches.shape[1]
    n_patches_per_row = ymax // patch_size
    
    complete_image = np.zeros((xmax, ymax, 3))
    
    row = -1
    col = 0
    for i in range(len(patches)):
        if i % n_patches_per_row == 0:
            row += 1
            col = 0
        complete_image[
        row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size,:
        ] = patches[i]
        col += 1
    return complete_image[0: target_shape[0], 0: target_shape[1], :]

def adjust_histogram(
    arr: np.ndarray, 
    in_ranges: list[tuple[int, int]],
    out_range: tuple[int, int]=(0,255), 
) -> np.ndarray:
    """ Adjust histogram of the image to have values in out_range.
    
    Args:
        arr (np.ndarray): input image [C, H, W]
        in_ranges (list[tuple[int, int]]): input range for each channel
        out_range (Tuple[int, int], optional): output range. Defaults to (0,255).
    
    Returns:
        np.ndarray: image with adjusted histogram
    """
 
    for c in range(arr.shape[0]):
        c_arr = arr[c, ...]
        c_min, c_max = in_ranges[c]
        if c_min == c_max:
            arr[c, ...] = np.zeros(c_arr.shape)
        else:
            c_arr = np.clip(c_arr, c_min, c_max)
            arr[c, ...] = (
                (c_arr - c_min) / (c_max - c_min) * 
                (out_range[1] - out_range[0]) + out_range[0]
            )
        
    return arr.astype(get_dtype_from_range(out_range))


def get_quantile_ranges(
    img: np.ndarray, 
    q: tuple[float] = (0.001, 0.999),
    mask_zero: bool = True,
    channel_last: bool = True
) -> list[tuple[float, float]]:
    """
    Get quantile ranges for each channel of the image.

    Args:
        img (np.ndarray): Input image array.
        q (tuple[float]): Quantiles to compute, ranging from 0 to 1. 
            Defaults to (0.001, 0.999).
        mask_zero (bool): Whether to ignore zero values in the computation. 
            Defaults to True.
        channel_last (bool): If True, assumes the last dimension of img represents channels. 
            Defaults to True.

    Returns:
        list[tuple[float, float]]: List of quantile ranges for each channel in the format (min, max).
    """
    # Transpose the image to have channels as the last dimension if channel_last is False
    if not channel_last:
        img = img.transpose(1, 2, 0)
    
    ranges = []  # List to store quantile ranges for each channel

    # Iterate over each channel in the image
    for c in range(img.shape[-1]):
        c_arr = img[:, :, c]  # Extract the channel data
        if mask_zero:
            c_arr = c_arr[c_arr != 0]  # Mask zero values if specified
        if c_arr.size == 0:
            c_min, c_max = 0, 0  # Handle empty channel case
        else:
            c_min, c_max = np.quantile(c_arr, q)  # Compute quantile values
        ranges.append((c_min, c_max))  # Append the quantile range to the list
    
    return ranges


def get_dtype_from_range(img_range: tuple[int, int]) -> np.ndarray:
    """ Reduce the dtype based on the input range.
    """
    if img_range[0] >= 0 and img_range[1] <= 255:
        return np.uint8
    elif img_range[0] >= 0 and img_range[1] <= 65535:
        return np.uint16
    else:
        return np.int16