import csv
from pathlib import Path
import yaml
import matplotlib.pyplot as plt


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
