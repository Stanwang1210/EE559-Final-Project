import argparse
import csv
import json
import logging
import random
from pathlib import Path

from tqdm import trange
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import MeanTimeClassifier, CNNClassifier
from dataset import IemocapMelDataset, collate_pad
from utils import (
    read_rows,
    split_rows,
    build_id_to_label,
    load_config,
    plot_training_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path("preprocessed/iemocap_melspec.h5"),
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("preprocessed/iemocap_labels.csv"),
    )
    parser.add_argument(
        "--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mel_nn.yaml"),
    )
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += y.size(0)

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def save_predictions(
    model: nn.Module,
    loader: DataLoader,
    rows: list[dict[str, str]],
    device: torch.device,
    id_to_label: dict[int, str],
    out_csv: Path,
) -> None:
    model.eval()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    write_rows: list[dict[str, object]] = []
    ptr = 0

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().tolist()
        true = y.cpu().tolist()

        for i in range(len(true)):
            row = rows[ptr + i]
            true_id = int(true[i])
            pred_id = int(pred[i])
            true_label_text = row.get("label", id_to_label.get(true_id, str(true_id)))
            pred_label_text = id_to_label.get(pred_id, str(pred_id))

            write_rows.append(
                {
                    "key": row["key"],
                    "split": row["split"],
                    "true_label": true_label_text,
                    "pred_label": pred_label_text,
                    "true_label_id": true_id,
                    "pred_label_id": pred_id,
                    "correct": int(true_id == pred_id),
                }
            )

        ptr += len(true)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "split",
                "true_label",
                "pred_label",
                "true_label_id",
                "pred_label_id",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(write_rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    batch_size = int(cfg.get("batch_size", 256))
    epochs = int(cfg.get("epochs", 100))
    lr = float(cfg.get("lr", 1e-4))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    model_type = str(cfg.get("model_type", "nn"))
    num_workers = int(cfg.get("num_workers", 8))
    weight_decay = float(cfg.get("weight_decay", 1e-5))

    set_seed(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.save_dir / "train.log"
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("train")
    logger.info("Training started")
    logger.info("Args: %s", vars(args))
    logger.info(
        "Config | model_type=%s batch_size=%d epochs=%d lr=%.6f hidden_dim=%d num_workers=%d",
        model_type,
        batch_size,
        epochs,
        lr,
        hidden_dim,
        num_workers,
    )

    h5_path = args.h5
    labels_csv = (
        args.labels_csv
        if args.labels_csv is not None
        else args.data_dir / "iemocap_labels.csv"
    )

    logger.info("Using HDF5: %s", h5_path)
    logger.info("Using labels CSV: %s", labels_csv)

    rows = read_rows(labels_csv)
    train_rows, valid_rows, test_rows = split_rows(rows)
    id_to_label = build_id_to_label(rows)

    num_classes = len({int(r["label_id"]) for r in rows})
    with h5py.File(h5_path, "r") as h5f:
        sample_key = train_rows[0]["key"]
        n_mels = int(h5f[sample_key].shape[0])

    train_ds = IemocapMelDataset(h5_path, train_rows)
    valid_ds = IemocapMelDataset(h5_path, valid_rows)
    test_ds = IemocapMelDataset(h5_path, test_rows)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_pad,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_pad,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_pad,
    )

    device = torch.device(args.device)
    if model_type == "cnn":
        model = CNNClassifier(
            n_mels=n_mels, hidden_dim=hidden_dim, num_classes=num_classes
        ).to(device)
    else:
        model = MeanTimeClassifier(
            n_mels=n_mels, hidden_dim=hidden_dim, num_classes=num_classes
        ).to(device)

    logger.info("Model initialized: %s", model)
    total_params = sum(p.numel() for p in model.parameters()) / 1024**2
    logger.info("Total parameters: %.2f MB", total_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_path = args.save_dir / "best_mean_time_model.pt"
    curve_path = args.save_dir / "training_curve.png"

    best_valid_acc = -1.0
    history = []

    pbar = trange(1, epochs + 1, desc="Training", leave=True)
    for epoch in pbar:
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * y.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_count += y.size(0)

        train_loss = train_loss_sum / train_count
        train_acc = train_correct / train_count
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
            }
        )
        pbar.set_postfix(val_loss=f"{valid_loss:.4f}", val_acc=f"{valid_acc:.4f}")

        logger.info(
            "Epoch %03d/%03d | train_loss=%.6f train_acc=%.6f | valid_loss=%.6f valid_acc=%.6f",
            epoch,
            epochs,
            train_loss,
            train_acc,
            valid_loss,
            valid_acc,
        )

        plot_training_curve(history, curve_path)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "n_mels": n_mels,
                    "hidden_dim": hidden_dim,
                    "num_classes": num_classes,
                    "best_valid_acc": best_valid_acc,
                },
                best_path,
            )
            logger.info(
                "New best valid acc=%.6f at epoch %d, saved model to %s",
                best_valid_acc,
                epoch,
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    logger.info("Best valid acc: %.6f", best_valid_acc)
    logger.info("Test loss: %.6f", test_loss)
    logger.info("Test acc: %.6f", test_acc)
    logger.info("Saved best model: %s", best_path)

    report = {
        "best_valid_acc": best_valid_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "num_classes": num_classes,
        "n_mels": n_mels,
        "history": history,
        "h5_path": str(h5_path),
        "labels_csv": str(labels_csv),
        "split_counts": {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "test": len(test_rows),
        },
    }
    report_path = args.save_dir / "train_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)
    logger.info("Saved report: %s", report_path)
    logger.info("Saved training curve: %s", curve_path)

    valid_pred_path = args.save_dir / "valid_predictions.csv"
    test_pred_path = args.save_dir / "test_predictions.csv"
    save_predictions(
        model, valid_loader, valid_rows, device, id_to_label, valid_pred_path
    )
    save_predictions(model, test_loader, test_rows, device, id_to_label, test_pred_path)
    logger.info("Saved valid predictions: %s", valid_pred_path)
    logger.info("Saved test predictions: %s", test_pred_path)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
