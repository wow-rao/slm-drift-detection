# Document Anomaly Detection - V2 (Chunk-Based Training)

## Quick Start

### Training

```bash
python src/exec/v2/train.py --epochs 50 --train_dir data/train
```

### Testing

```bash
python src/exec/v2/play.py --test_dir data/test
```

## Example Usage

### Training with Custom Parameters

```bash
python src/exec/v2/train.py \
    --epochs 100 \
    --train_dir data/train \
    --batch_size 16 \
    --lr 0.0005 \
    --log_dir logs_v2 \
    --viz_freq 10
```

### Testing with Custom Model

```bash
python src/exec/v2/play.py \
    --test_dir data/test \
    --model_path logs_v2/model.pt \
    --threshold_path logs_v2/threshold.npy \
    --output_file results_v2.txt
```

## Interpreting Results

- **Reconstruction Error (normalized)**: Average reconstruction loss per chunk
- **Number of Chunks**: How many segments the document was split into
- **Anomaly Likelihood**: 
  - < 100% = Within normal distribution
  - > 100% = Anomalous
  - > 200% = Highly anomalous

## Monitoring Training

While training is running, view the live loss plot:

```bash
watch -n 2 'open logs/training_loss.png'
watch -n 2 'eog logs/training_loss.png'
```
