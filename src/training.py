import torch
import pandas as pd

from .data import build_loaders
from .models import build_model
from .evaluation import (
    validate_one_epoch,
    evaluate_patients_3d,
    summarize_evaluation,
)
bce_criterion = torch.nn.BCEWithLogitsLoss()

def soft_dice_loss(logits, targets, eps=1e-5):
    p = torch.sigmoid(logits)

    num = 2*(p*targets).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps

    return 1 - (num/den).mean()


def soft_dice_loss_batch(logits, targets, eps=1e-5):
    p = torch.sigmoid(logits)

    num = 2*(p*targets).sum() + eps
    den = p.sum() + targets.sum() + eps

    return 1 - (num/den)

    
def bce_dice_batch_loss(logits, targets):
    bce = bce_criterion(logits, targets)
    soft_dice = soft_dice_loss_batch(logits, targets)

    return bce + soft_dice

    
def bce_dice_loss(logits, targets):
    bce = bce_criterion(logits, targets)
    soft_dice = soft_dice_loss(logits, targets)

    return bce + soft_dice


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n_samples = 0

    for images, masks in loader:
        
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / n_samples


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    seed,
    run_name,
    history,
    config,
    best_score,
    best_epoch,
):
    torch.save(
        {
            "epoch": epoch,
            "seed": seed,
            "run_name": run_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config.copy(),
            "preprocessing": config["preprocessing"],
            "best_score": best_score,
            "best_epoch": best_epoch
        },
        path
    )

def run_experiments(config, project_root, train_dataset, val_dataset):
    results = []
    output_dir = project_root / "outputs"
    checkpoint_dir = output_dir / "checkpoints"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config["run_training"]:
        for seed in config["seeds"]:
            run_name = f"{config['experiment_name']}_seed{seed}"
            paths = {
                "best": checkpoint_dir / f"{run_name}_best.pt",
                "last": checkpoint_dir / f"{run_name}_last.pt",
                "history": output_dir / f"{run_name}_history.csv",
                "val_metrics": output_dir / f"{run_name}_val_metrics.csv",
                "val_patients": output_dir / f"{run_name}_val_patients.csv"
            }
            
            for path in paths.values():
                if path.exists():
                    raise FileExistsError(f"Plik już istnieje: {path}")
    
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            train_loader, val_loader = build_loaders(
                train_dataset,
                val_dataset, 
                batch_size=config["batch_size"],
                seed = seed,
                sampling = config["sampling"]
            )
        
            print(f"\nSeed {seed}: train {len(train_loader)} paczki, val {len(val_loader)} paczki")
        
            torch.manual_seed(seed)
            model = build_model(config["model"]).to(device)
        
            if config["optimizer"] == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            else:
                raise ValueError("nieznany optimizer")
        
            if config["loss"] == "bce_dice":
                criterion = bce_dice_loss
            elif config["loss"] == "bce_dice_batch":
                criterion = bce_dice_batch_loss
            else:
                raise ValueError("nieznany loss")
        
            print("run name: ", run_name)
    
            history = []
            best_score = -float("inf")
            best_epoch = None
    
            for epoch in range(1, config["num_epochs"] + 1):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_metrics = validate_one_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    threshold=config["threshold"]
                )

                if config["selection_metric"] == "dice_3d":
                    patient_results, _, _ = evaluate_patients_3d(
                        model,
                        val_dataset.pairs_df,
                        project_root,
                        device,
                        patient_params=val_dataset.patient_params,
                        batch_size=config["batch_size"],
                        threshold=config["threshold"],
                    )

                    val_metrics["dice_3d"] = float(patient_results["dice_3d"].mean())
                    val_metrics["dice_3d_filtered"] = float(
                        patient_results["dice_3d_filtered"].mean()
                    )
    
                history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **val_metrics,
                })
    
                current_score = val_metrics[config["selection_metric"]]
                if current_score > best_score:
                    best_score = current_score
                    best_epoch = epoch
    
                    save_checkpoint(
                        paths["best"],
                        model,
                        optimizer,
                        epoch,
                        seed,
                        run_name,
                        history,
                        config,
                        best_score,
                        best_epoch
                    )
    
                print("\nepoka: ", epoch)
                print("train loss: ", train_loss)
                print("val metrics: ", val_metrics)
    
            print("\nKoniec treningu")
            pd.DataFrame(history).to_csv(paths["history"], index=False)
            
            if config["save_last"]:
                    save_checkpoint(
                        paths["last"],
                        model,
                        optimizer,
                        epoch,
                        seed,
                        run_name,
                        history,
                        config,
                        best_score,
                        best_epoch
                    )
    
            print("best epoch: ", best_epoch)
            print("best score: ", best_score)
    
            checkpoint = torch.load(
                paths["best"], map_location="cpu", weights_only=True
            )
    
            evaluation_model = build_model(checkpoint["config"]["model"])
            evaluation_model.load_state_dict(checkpoint["model_state_dict"])
    
            patient_results, dice_scores, has_annotation = evaluate_patients_3d(
                evaluation_model,
                val_dataset.pairs_df,
                project_root,
                torch.device("cpu"),
                patient_params=val_dataset.patient_params,
                batch_size=config["batch_size"],
                threshold=config["threshold"]
            )
    
            summary = summarize_evaluation(patient_results, dice_scores, has_annotation)
    
            print("epoka checkpointu: ", checkpoint["epoch"])
            print("summary: ", summary)
    
            run_result = {
                "run_name": checkpoint["run_name"],
                "seed": checkpoint["seed"],
                "epoch": checkpoint["epoch"],
                **summary,
            }
            
            pd.DataFrame([run_result]).to_csv(paths["val_metrics"], index=False)
            patient_results.to_csv(paths["val_patients"], index=False)

            results.append(run_result)
    
            print("ścieżka val_metrics: ", paths["val_metrics"])
            print("ścieżka val_patients: ", paths["val_patients"])
                    
    
    else:
        print("Trening wyłączony w konfiguracji")

    return pd.DataFrame(results)

