import torch
import numpy as np
from scipy import ndimage
import pandas as pd
from torch.utils.data import DataLoader
from .data import MRIDataset

def dice_per_sample(predictions, targets):
    targets_logic = targets > 0

    targets_sum = targets_logic.sum(dim=(1, 2, 3))
    predictions_sum = predictions.sum(dim=(1, 2, 3))
    intersection = (targets_logic & predictions).sum(dim=(1, 2, 3))

    denominator = targets_sum + predictions_sum

    return torch.where(
        denominator == 0,
        1.0,
        2 * intersection / denominator.clamp_min(1)
    )

def dice_3d(predictions, targets):
    targets = targets > 0
    intersection = (predictions & targets).sum().item()
    pred_count = predictions.sum().item()
    target_count = targets.sum().item()

    denominator = pred_count + target_count

    dice = 2 * intersection / denominator if denominator != 0 else 1.0

    return dice

def keep_largest_component(predictions):
    predictions = predictions.squeeze(1)
    structure = ndimage.generate_binary_structure(3, 1)
    labeled, n_labels = ndimage.label(predictions, structure=structure)
    
    if n_labels == 0:
        return predictions.unsqueeze(1)
        
    sizes = np.bincount(labeled.ravel())
    largest_label = sizes[1:].argmax() + 1
    return torch.from_numpy(labeled == largest_label).unsqueeze(1)


def validate_one_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    n_samples = 0
    n_annotated = 0
    n_empty = 0
    total_loss = 0.0
    total_dice = 0.0
    total_dice_annotated = 0.0
    total_dice_empty = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            predictions = torch.sigmoid(logits) > threshold
            dice = dice_per_sample(predictions, masks)
            has_annotation = masks.sum(dim=(1, 2, 3)) > 0
            batch_size = images.shape[0]
            
            n_samples += batch_size
            n_annotated += has_annotation.sum().item()
            n_empty += (~has_annotation).sum().item()
            
            total_loss += loss.item() * batch_size
            total_dice += dice.sum().item()
            total_dice_annotated += dice[has_annotation].sum().item()
            total_dice_empty += dice[~has_annotation].sum().item()

    return {
        "loss": total_loss / n_samples,
        "dice": total_dice / n_samples,
        "dice_annotated": total_dice_annotated / n_annotated,
        "dice_empty" : total_dice_empty / n_empty
    }

def predict_patient(model, patient_loader, device, threshold=0.5):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for images, masks in patient_loader:
            images = images.to(device)
            logits = model(images)
            
            preds_batch = (torch.sigmoid(logits) > threshold).cpu()
            
            predictions.append(preds_batch)
            targets.append((masks > 0).cpu())

    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)

    return predictions, targets
    
            
def evaluate_patients_3d(
    model,
    split_df,
    project_root,
    device,
    patient_params=None,
    batch_size=4,
    threshold=0.5,
):
    patient_results = []
    dice_batches = []
    annotation_batches = []

    for patient_id, patient_df in split_df.groupby("patient_id"):
        patient_df = patient_df.sort_values("slice_id")
        
        patient_dataset = MRIDataset(patient_df, project_root, patient_params)
        patient_loader = DataLoader(
            patient_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        predictions, targets = predict_patient(
            model, patient_loader, device, threshold = threshold
        )
        
        dice_batches.append(dice_per_sample(predictions, targets))
        annotation_batches.append(targets.flatten(1).any(dim=1))

        dice3d = dice_3d(predictions, targets)
        
        filtered_predictions = keep_largest_component(predictions)
        dice3d_filtered = dice_3d(filtered_predictions, targets)

        patient_results.append({
            "patient_id": patient_id,
            "dice_3d": dice3d,
            "dice_3d_filtered": dice3d_filtered,
        })

    return (
        pd.DataFrame(patient_results),
        torch.cat(dice_batches),
        torch.cat(annotation_batches),
    )


def summarize_evaluation(patient_results, dice_scores, has_annotation):
    dice = dice_scores.mean().item()
    dice_annotated = dice_scores[has_annotation].mean().item()
    dice_empty = dice_scores[~has_annotation].mean().item()
    dice3d = float(patient_results["dice_3d"].mean())
    dice3d_filtered = float(patient_results["dice_3d_filtered"].mean())

    return {
        "dice": dice,
        "dice_annotated": dice_annotated,
        "dice_empty": dice_empty,
        "dice_3d": dice3d,
        "dice_3d_filtered": dice3d_filtered,
    }

def summarize_runs(results_df):
    if results_df.empty:
        return pd.DataFrame(columns=["mean", "std"])
    
    labels = [
        "dice",
        "dice_annotated",
        "dice_empty",
        "dice_3d",
        "dice_3d_filtered",
    ]
    
    summary = []

    for label in labels:
        col = results_df[label]

        summary.append({
            "metric": label,
            "mean": col.mean(),
            "std": col.std(ddof=1)
        })

    return pd.DataFrame(summary).set_index("metric")
        