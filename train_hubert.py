#!/usr/bin/env python3
"""Train a CNN classifier on IEMOCAP HuBERT-base features.

Model behavior:
- Input feature shape: [B, hidden_dim, T]
- 1D CNN over time axis T
- Global mean pooling + linear classifier
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import trange, tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train classifier on HuBERT features"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("preprocessed/hubert-base")
    )
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path("preprocessed/hubert-base/iemocap_hubert-base.h5"),
        help="Path to iemocap_hubert-base.h5",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("preprocessed/hubert-base/iemocap_labels.csv"),
        help="Path to iemocap_labels.csv",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hubert-base_cnn.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/hubert_cnn"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_coeff_mean",
        action="store_true",
        help="Use haar mean coeff features instead of HuBERT features. If set, `--h5` should point to the mean coeff .npy file.",
        default=False,
    )
    parser.add_argument(
        "--coeff_path",
        type=Path,
        help="Path to the OMP coeff .npz file.",
    )
    parser.add_argument(
        "--dict_path",
        type=Path,
        help="Path to the OMP dict .npy file.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Config must be a mapping: {config_path}")
    return cfg


class HuBERTCNNClassifier(nn.Module):
    def __init__(
        self,
        n_mels: int,
        hidden_dim: int,
        num_classes: int,
        num_encoder_layers: int,
        temporal_pool_kernel: int = 2,
    ) -> None:
        super().__init__()
        d_model = hidden_dim * 2
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.MaxPool1d(kernel_size=temporal_pool_kernel, stride=temporal_pool_kernel),
            nn.Conv1d(d_model, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.MaxPool1d(kernel_size=temporal_pool_kernel, stride=temporal_pool_kernel),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.net = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_mels, T]
        x = self.cnn(x)  # [B, d_model, T]
        x = self.net(x.transpose(1, 2))  # [B, T, d_model]
        x_pred = self.fc(x).mean(dim=1)  # [B, num_classes]
        return x_pred
    
class HuBERTClassifier(nn.Module):
    def __init__(
        self,
        n_mels: int,
        num_classes: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_mels, n_mels//2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(n_mels//2, n_mels//4),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(n_mels//4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_mels, T]
          # [B, T, d_model]
        x_pred = self.fc(x.squeeze(-1))  # [B, num_classes]
        return x_pred


class IemocapHubertDataset(Dataset):
    def __init__(self, h5_path: Path, rows: list[dict[str, str]]) -> None:
        self.h5_path = h5_path
        self.rows = rows
        self._h5 = None

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        h5f = self._get_h5()
        feat = h5f[row["key"]][()]
        x = torch.tensor(feat, dtype=torch.float32)
        y = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return x, y


class IemocapHubertCoeffDataset(Dataset):
    def __init__(
        self,
        coeff_npz_path: Path,
        dictionary_path: Path,
        rows: list[dict[str, str]] | None = None,
    ) -> None:
        data = np.load(coeff_npz_path)
        coeffs = np.asarray(data["coeffs"], dtype=np.float64)
        keys = np.asarray(data["keys"]).astype(str)
        label_ids = np.asarray(data["label_id"], dtype=np.int64)

        if len(keys) != coeffs.shape[0] or len(label_ids) != coeffs.shape[0]:
            raise RuntimeError("`coeffs`, `keys`, and `label_id` must have same length.")

        dictionary = np.asarray(np.load(dictionary_path), dtype=np.float64)
        if dictionary.ndim != 2:
            raise RuntimeError(f"Dictionary must be 2-D, got shape={dictionary.shape}")

        if dictionary.shape[1] != coeffs.shape[1]:
            if dictionary.shape[0] == coeffs.shape[1]:
                dictionary = dictionary.T
            else:
                raise RuntimeError(
                    "Dictionary atom dimension does not match coeff dimension: "
                    f"dict={dictionary.shape}, coeffs={coeffs.shape}"
                )

        self.feats = [dictionary @ coeff for coeff in tqdm(coeffs, desc="Computing features from coeffs")]
        self.coeffs = coeffs
        self.keys = keys
        self.label_ids = label_ids
        self.dictionary = dictionary

    def __len__(self) -> int:
        return self.coeffs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        feat = self.feats[idx]
        x = torch.tensor(feat[:, None], dtype=torch.float32)
        y = torch.tensor(int(self.label_ids[idx]), dtype=torch.long)
        return x, y


def collate_pad(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    n_feats = xs[0].shape[0]
    max_t = max(x.shape[1] for x in xs)

    x_pad = torch.zeros(len(xs), n_feats, max_t, dtype=torch.float32)
    for i, x in enumerate(xs):
        t = x.shape[1]
        x_pad[i, :, :t] = x
    y = torch.stack(ys)
    return x_pad, y


def read_rows(labels_csv: Path) -> list[dict[str, str]]:
    with labels_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"key", "split", "label_id"}
    if not rows:
        raise RuntimeError(f"No rows found in {labels_csv}")
    if not required.issubset(set(rows[0].keys())):
        raise RuntimeError(f"CSV missing required columns: {required}")
    return rows


def split_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    train_rows = [r for r in rows if r["split"] == "train"]
    valid_rows = [r for r in rows if r["split"] == "valid"]
    test_rows = [r for r in rows if r["split"] == "test"]

    if not train_rows or not valid_rows or not test_rows:
        raise RuntimeError(
            "Split rows are empty. Ensure preprocessing used split rule: "
            "Session1-3=train, Session4=valid, Session5=test."
        )
    return train_rows, valid_rows, test_rows


def build_id_to_label(rows: list[dict[str, str]]) -> dict[int, str]:
    id_to_label: dict[int, str] = {}
    for r in rows:
        if "label" not in r or r["label"] == "":
            continue
        lid = int(r["label_id"])
        if lid not in id_to_label:
            id_to_label[lid] = r["label"]
    return id_to_label


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


def plot_training_curve(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        raise RuntimeError("History is empty, cannot plot training curve.")

    epochs = [int(h["epoch"]) for h in history]
    train_loss = [float(h["train_loss"]) for h in history]
    valid_loss = [float(h["valid_loss"]) for h in history]
    train_acc = [float(h["train_acc"]) for h in history]
    valid_acc = [float(h["valid_acc"]) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, marker="o", label="train_loss")
    axes[0].plot(epochs, valid_loss, marker="o", label="valid_loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="train_acc")
    axes[1].plot(epochs, valid_acc, marker="o", label="valid_acc")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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
    weight_decay = float(cfg.get("weight_decay", 1e-5))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    step_updates = int(cfg.get("step_updates", 1))
    num_workers = int(cfg.get("num_workers", 8))
    num_encoder_layers = int(cfg.get("num_encoder_layers", 2))
    temporal_pool_kernel = int(cfg.get("temporal_pool_kernel", 2))
    set_seed(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.save_dir / "train.log"
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("train_hubert")
    logger.info("Training started")
    logger.info("Args: %s", vars(args))
    logger.info(
        "Config | model_type=cnn batch_size=%d epochs=%d lr=%.6f weight_decay=%.6f hidden_dim=%d num_workers=%d",
        batch_size,
        epochs,
        lr,
        weight_decay,
        hidden_dim,
        num_workers,
    )
    use_mean_data = 'mean' in args.data_dir.name
    logger.info("Using mean features: %s", use_mean_data)
    

    labels_csv = (
        args.labels_csv
        if args.labels_csv is not None
        else args.data_dir / "iemocap_labels.csv"
    )
    h5_path = (
        args.h5 if args.h5 is not None else args.data_dir / "iemocap_hubert-base.h5"
    )
    logger.info("Using HDF5: %s", h5_path)
    rows = read_rows(labels_csv)
    logger.info("Using labels CSV: %s", labels_csv)
    num_classes = len({int(r["label_id"]) for r in rows})
    train_rows, valid_rows, test_rows = split_rows(rows)
    id_to_label = build_id_to_label(rows)
    n_feats = 768
    if args.use_coeff_mean:
        if args.coeff_path is None or args.dict_path is None:
            raise RuntimeError("OMP coeff and dict paths must be provided when using mean features.")
        
        logger.info("Using OMP coeffs from: %s", args.coeff_path)
        logger.info("Using OMP dictionary from: %s", args.dict_path)
        train_ds = IemocapHubertCoeffDataset(f'{args.coeff_path}_train_coffs.npz', f'{args.dict_path}_dictionary.npy')
        valid_ds = IemocapHubertCoeffDataset(f'{args.coeff_path}_valid_coffs.npz', f'{args.dict_path}_dictionary.npy')
        test_ds = IemocapHubertCoeffDataset(f'{args.coeff_path}_test_coffs.npz', f'{args.dict_path}_dictionary.npy')

    else:
        
        
        train_ds = IemocapHubertDataset(h5_path, train_rows)
        valid_ds = IemocapHubertDataset(h5_path, valid_rows)
        test_ds = IemocapHubertDataset(h5_path, test_rows)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=None if use_mean_data else collate_pad,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=None if use_mean_data else collate_pad,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=None if use_mean_data else collate_pad,
    )

    device = torch.device(args.device)
    if use_mean_data:
        MODEL_CLASS = HuBERTClassifier
    else:
        MODEL_CLASS = HuBERTCNNClassifier
        
    model = MODEL_CLASS(
        n_mels=n_feats,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_encoder_layers=num_encoder_layers,
        temporal_pool_kernel=temporal_pool_kernel,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    logger.info("Model initialized: %s", model)
    # 1) Parameter count in millions
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total parameters: %.2f M", total_params / 1e6)

    best_path = args.save_dir / "best_hubert_cnn_model.pt"
    curve_path = args.save_dir / "training_curve.png"

    best_valid_acc = -1.0
    history = []

    pbar = trange(1, epochs + 1, desc="Training", leave=True)
    for epoch in pbar:
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        step = 0
        optimizer.zero_grad()
        for x, y in train_loader:
            step += 1
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if step % step_updates == 0:
                optimizer.step()
                optimizer.zero_grad()

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
                    "n_feats": n_feats,
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
        "n_feats": n_feats,
        "history": history,
        "h5_path": str(h5_path) ,
        "coeff_path": str(args.coeff_path) if args.coeff_path else None,
        "dict_path": str(args.dict_path) if args.dict_path else None,
        "use_coeff_mean": args.use_coeff_mean,
        "use_mean_data": use_mean_data,
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
