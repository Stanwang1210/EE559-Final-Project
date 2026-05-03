#!/usr/bin/env python3
"""Preprocess IEMOCAP utterance audio into mel-spectrograms and label files.

This script:
1) Parses utterance-level labels from Session*/dialog/EmoEvaluation/*.txt
2) Converts utterance wav files in Session*/sentences/wav/**/*.wav to mel-spectrograms
3) Stores all spectrograms in one HDF5 file for fast loading
4) Exports key-label mappings as CSV + JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import h5py
import librosa
import numpy as np
from tqdm import tqdm


LABEL_LINE_RE = re.compile(r"^\[[^\]]+\]\s+(\S+)\s+([a-z]{3})\s+\[[^\]]+\]\s*$")
REMOVED_DATA_LABELS = {"xxx", "dis", "oth", "fea", "sur"}


def split_from_key(key: str) -> str:
    m = re.match(r"^Ses(\d{2})", key)
    if not m:
        return "unknown"
    session_id = int(m.group(1))
    if 1 <= session_id <= 3:
        return "train"
    if session_id == 4:
        return "valid"
    if session_id == 5:
        return "test"
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("IEMOCAP_full_release"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("preprocessed"),
    )
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--win_length", type=int, default=1024)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument(
        "--n-mfcc", type=int, default=40, help="Number of MFCC coefficients"
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["mel", "mfcc"],
        default="mel",
    )
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
    )
    return parser.parse_args()


def parse_emotion_labels(iemocap_root: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    conflicts = 0

    eval_files = sorted(iemocap_root.glob("Session*/dialog/EmoEvaluation/*.txt"))
    eval_files = [p for p in eval_files if not p.name.startswith("._")]

    for file_path in eval_files:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = LABEL_LINE_RE.match(line.strip())
                if not m:
                    continue
                key, emo = m.group(1), m.group(2)
                if key in labels and labels[key] != emo:
                    conflicts += 1
                    continue
                labels[key] = emo


    return labels


def list_utterance_wavs(iemocap_root: Path) -> list[Path]:
    wavs = sorted(iemocap_root.glob("Session*/sentences/wav/**/*.wav"))
    wavs = [p for p in wavs if not p.name.startswith("._")]
    return wavs


def main() -> None:
    args = parse_args()
    root = args.root

    if not root.exists():
        raise FileNotFoundError(f"IEMOCAP root not found: {root}")

    args.output_dir = args.output_dir / f"{args.feature_type}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    h5_name = f"iemocap_{args.feature_type}.h5"
    h5_path = args.output_dir / h5_name
    csv_path = args.output_dir / "iemocap_labels.csv"
    json_path = args.output_dir / "iemocap_labels.json"
    meta_path = args.output_dir / "iemocap_meta.json"

    labels = parse_emotion_labels(root)
    wav_files = list_utterance_wavs(root)

    labels = {k: v for k, v in labels.items() if v not in REMOVED_DATA_LABELS}

    available_labels = sorted(set(labels.values()))
    label_to_id = {lab: i for i, lab in enumerate(available_labels)}

    rows: list[dict[str, object]] = []
    missing_label = 0
    duplicate_key = 0
    seen_keys: set[str] = set()
    feature_shapes: Counter[tuple[int, int]] = Counter()
    split_distribution: Counter[str] = Counter()

    fmax = args.fmax if args.fmax is not None else args.sr / 2.0

    with h5py.File(h5_path, "w") as h5f:
        for wav_path in tqdm(
            wav_files, desc=f"Extracting {args.feature_type} features"
        ):
            key = wav_path.stem
            split = split_from_key(key)

            if key in seen_keys:
                duplicate_key += 1
                continue
            seen_keys.add(key)

            label = labels.get(key)
            if label is None:
                missing_label += 1
                continue

            y, _ = librosa.load(wav_path, sr=args.sr, mono=True)
            if args.feature_type == "mel":
                feat = librosa.feature.melspectrogram(
                    y=y,
                    sr=args.sr,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    win_length=args.win_length,
                    n_mels=args.n_mels,
                    fmin=args.fmin,
                    fmax=fmax,
                    power=2.0,
                )
                feat = librosa.power_to_db(feat, ref=np.max)
            else:
                feat = librosa.feature.mfcc(
                    y=y,
                    sr=args.sr,
                    n_mfcc=args.n_mfcc,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    win_length=args.win_length,
                    fmin=args.fmin,
                    fmax=fmax,
                )
            feat = feat.astype(np.float32)

            ds = h5f.create_dataset(key, data=feat, compression="gzip")
            ds.attrs["label"] = label
            ds.attrs["label_id"] = int(label_to_id[label])
            ds.attrs["split"] = split
            ds.attrs["wav_path"] = str(wav_path)
            ds.attrs["feature_type"] = args.feature_type

            feature_shapes[tuple(feat.shape)] += 1
            split_distribution[split] += 1
            rows.append(
                {
                    "key": key,
                    "split": split,
                    "label": label,
                    "label_id": label_to_id[label],
                    "num_features": feat.shape[0],
                    "num_frames": feat.shape[1],
                    "wav_path": str(wav_path),
                }
            )

    rows.sort(key=lambda x: x["key"])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "split",
                "label",
                "label_id",
                "num_features",
                "num_frames",
                "wav_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    label_json = {
        "label_to_id": label_to_id,
        "id_to_label": {str(v): k for k, v in label_to_id.items()},
        "split_rule": {"train": [1, 2, 3], "valid": [4], "test": [5]},
        "items": [
            {
                "key": r["key"],
                "split": r["split"],
                "label": r["label"],
                "label_id": r["label_id"],
            }
            for r in rows
        ],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(label_json, f, ensure_ascii=True, indent=2)

    meta = {
        "root": str(root),
        "feature_type": args.feature_type,
        "split_rule": {"train": [1, 2, 3], "valid": [4], "test": [5]},
        "num_wavs_found": len(wav_files),
        "num_labels_found": len(labels),
        "num_saved": len(rows),
        "num_missing_label": missing_label,
        "num_duplicate_key": duplicate_key,
        "split_distribution": dict(split_distribution),
        "audio": {
            "sr": args.sr,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "win_length": args.win_length,
            "n_mels": args.n_mels,
            "n_mfcc": args.n_mfcc,
            "fmin": args.fmin,
            "fmax": fmax,
        },
        "label_distribution": dict(Counter([r["label"] for r in rows])),
        "top_feature_shapes": [
            [list(shape), cnt] for shape, cnt in feature_shapes.most_common(10)
        ],
        "files": {
            "h5": str(h5_path),
            "labels_csv": str(csv_path),
            "labels_json": str(json_path),
        },
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    print(f"Saved spectrogram file: {h5_path}")
    print(f"Saved label CSV:       {csv_path}")
    print(f"Saved label JSON:      {json_path}")
    print(f"Saved metadata JSON:   {meta_path}")
    print(f"Total saved samples:   {len(rows)}")


if __name__ == "__main__":
    main()
