import numpy as np
import torch
from scipy import ndimage


def rotate_pair(image, mask, angle, image_fill):
    image_array = image.numpy()
    mask_array = mask.numpy()

    fill = np.asarray(image_fill, dtype=np.float32).reshape(-1, 1, 1)

    shifted_image = image_array - fill

    rotated_image = ndimage.rotate(
        shifted_image,
        angle,
        axes=(1, 2),
        reshape=False,
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    ) + fill

    rotated_mask = ndimage.rotate(
        mask_array,
        angle,
        axes=(1, 2),
        reshape=False,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )

    return (
        torch.from_numpy(rotated_image),
        torch.from_numpy(rotated_mask),
    )


class RandomRotation:
    def __init__(self, max_angle, p):
        self.max_angle = max_angle
        self.p = p

    def __call__(self, image, mask, image_fill):
        rnd_num = torch.rand(()).item()

        if rnd_num >= self.p:
            return image, mask

        angle = torch.empty(()).uniform_(
            -self.max_angle, self.max_angle
        ).item()
    
        return rotate_pair(image, mask, angle, image_fill)