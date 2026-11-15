import numpy as np
import torch

def preprocess_pair(image, mask, patient_stats=None, percentile_bounds=None):
    image = image.astype(np.float32) / 255

    if percentile_bounds is not None:
        lower, upper = percentile_bounds
        image = rescale_percentiles(image, lower, upper)
    
    if patient_stats is not None:
        mean, std = patient_stats
        image = ((image - mean) / std).astype(np.float32)

    image_tensor = torch.from_numpy(image).permute(2, 0, 1)
    mask_tensor = torch.from_numpy(mask > 0).float().unsqueeze(0)

    return (image_tensor, mask_tensor)


def compute_patient_stats(volume, percentile_bounds=None):
    volume = volume.astype(np.float32) / 255

    if percentile_bounds is not None:
        lower, upper = percentile_bounds
        volume = rescale_percentiles(volume, lower, upper)

    mean = volume.mean(axis=(0, 1, 2), dtype=np.float64)
    std = volume.std(axis=(0, 1, 2), dtype=np.float64)

    return (mean, std)


def rescale_percentiles(array, lower, upper):
    if upper <= lower:
        raise ValueError("upper musi być większe niż lower")

    scaled = (array - lower) / (upper - lower)
    clipped = np.clip(scaled, 0, 1)

    return clipped.astype(np.float32)

def compute_percentile_bounds(
    volume, lower_percentile=10, upper_percentile=99
):
    if not (0 <= lower_percentile < upper_percentile <= 100):
        raise ValueError("niepoprawne wartości percentyli")

    volume = (volume.astype(np.float32) / 255)
    lower, upper = np.percentile(volume, [lower_percentile, upper_percentile])

    return lower, upper