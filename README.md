# EE559 Final Project

Speech Emotion Recognition on the IEMOCAP dataset.

## Dataset

IEMOCAP is not publicly mirrored in this repository. Please request/download it from:

- https://sail.usc.edu/iemocap/iemocap_release.htm

After extraction, place the dataset folder as:

```bash
IEMOCAP_full_release/
```

at the project root.

## Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the main pipeline script:

```bash
bash run.sh
```

## Data Processing

The preprocessing step extracts acoustic features (Mel spectrogram and MFCC) and writes outputs under `preprocessed/`.

```bash
feature_type=("mel" "mfcc")
output_dir="preprocessed"

for ft in "${feature_type[@]}"; do
    echo "Processing feature type: $ft"
    python data_preprocess.py \
        --root IEMOCAP_full_release \
        --output_dir "${output_dir}" \
        --feature_type "${ft}" &
done

wait
```

Expected outputs include:

- `preprocessed/mel/iemocap_mel.h5` (or mel-spectrogram variant)
- `preprocessed/mfcc/iemocap_mfcc.h5`
- `preprocessed/*/iemocap_labels.csv`

## Model Training and Inference

Training is driven by feature/model-specific config files in `configs/`.

```bash
for ft in "${feature_type[@]}"; do
    for mt in "${model_type[@]}"; do
        echo "Training model for feature type: $ft and model type: $mt"
        CUDA_VISIBLE_DEVICES=${cuda_idx} python train.py \
            --config "configs/${ft}_${mt}.yaml" \
            --data-dir "${output_dir}/${ft}" \
            --h5 "${output_dir}/${ft}/iemocap_${ft}.h5" \
            --labels-csv "${output_dir}/${ft}/iemocap_labels.csv" \
            --device "cuda:0" \
            --save-dir "exp/${ft}_${mt}"
    done
done
```

Artifacts are saved to `exp/<feature>_<model>/`, including logs, checkpoints, and prediction reports.

## Configs

See the `configs/` directory for full experiment settings (for example: `configs/mel_cnn.yaml`, `configs/mfcc_cnn.yaml`, `configs/mel_nn.yaml`, `configs/mfcc_nn.yaml`).

Typical settings:

- `batch_size: 256`
- `epochs: 50`
- `lr: 1e-4`
- `weight_decay: 1e-5`
- `num_workers: 8`
- `seed: 42`
- `model_type: cnn`
- `step_updates: 1`
- `hidden_dim: 256`
