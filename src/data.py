import numpy as np
import tifffile
from .preprocessing import compute_patient_stats, preprocess_pair, compute_percentile_bounds
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader


def load_patient_volume(patient_df, project_root):
    patient_df = patient_df.sort_values("slice_id")

    volume = np.stack([
        tifffile.imread(project_root / path)
        for path in patient_df["image_path"]
    ])

    return volume


def compute_sampling_weights(pairs_df):
    mean = pairs_df.groupby("patient_id")["foreground_pixels"].transform("mean")
    sum_area = pairs_df.groupby("patient_id")["foreground_pixels"].transform("sum")

    return ( (pairs_df["foreground_pixels"] + 0.1 * mean) 
                / (1.1 * sum_area) ).to_numpy(copy=True)


def build_weighted_sampler(pairs_df, seed):
    weights = compute_sampling_weights(pairs_df)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=pairs_df.shape[0],
        replacement=True,
        generator=generator
    )

    return sampler
    

class MRIDataset(Dataset):
    def __init__(self, pairs_df, project_root, patient_params=None, transform=None):
        self.pairs_df = pairs_df
        self.project_root = project_root
        self.patient_params = patient_params
        self.transform = transform

    def __len__(self):
        return self.pairs_df.shape[0]

    def __getitem__(self, index):
        row = self.pairs_df.iloc[index]
        image = tifffile.imread(self.project_root / row["image_path"])
        mask = tifffile.imread(self.project_root / row["mask_path"])

        bounds = None
        stats = None
        
        if self.patient_params is not None:
            params = self.patient_params[row["patient_id"]]
            stats = (params["mean"], params["std"])
            bounds = params["percentile_bounds"]

        image, mask = preprocess_pair(
            image,
            mask,
            patient_stats=stats,
            percentile_bounds=bounds,
        )

        if self.transform is not None:
            return self.transform(image, mask)
            
        return image, mask


def build_loaders(
    train_dataset, val_dataset, batch_size, seed, sampling
):
    if sampling == "patient_area":
        sampler = build_weighted_sampler(train_dataset.pairs_df, seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler = sampler
        )
    elif sampling == "shuffle":
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle = True,
            generator=generator
        )
    else:
        raise ValueError("nieznany sampling")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


def build_patient_params(pairs_df, project_root, percentiles=None):
    params = {}

    for patient_id, patient_df in pairs_df.groupby("patient_id"):
        raw_volume = load_patient_volume(patient_df, project_root)
        bounds = None
        
        if percentiles is not None:
            bounds = compute_percentile_bounds(raw_volume, *percentiles)
    
        mean, std = compute_patient_stats(raw_volume, percentile_bounds=bounds)

        params[patient_id] = {
            "mean": mean,
            "std": std,
            "percentile_bounds": bounds,
        }

    return params

    
    