# Document Anomaly Detection System

A modular system for detecting anomalous documents using BERT embeddings and autoencoder neural networks.

## Quick Start

### 1. Setup

```bash
pip install -r requirements.txt
python setup_check.py
```

### 2. Prepare Data

```bash
mkdir -p data/train data/test
```

### 3. Train

```bash
python src/exec/v1/train.py --epochs 50 --train_dir data/train
```

### 4. Test

```bash
python src/exec/v1/play.py --test_dir data/test
```

## Documentation

- **[Complete Usage Guide](src/exec/v1/README.md)**
- **[Project Structure](PROJECT_STRUCTURE.md)**

## Example Usage

### Training with Custom Parameters

```bash
python src/exec/v1/train.py \
    --epochs 100 \
    --train_dir data/train \
    --batch_size 16 \
    --lr 0.0005 \
    --viz_freq 10
```

### Testing with Custom Model

```bash
python src/exec/v1/play.py \
    --test_dir data/test \
    --model_path logs/model.pt \
    --output_file my_results.txt
```

### Monitoring Training

While training is running, view the live loss plot:

```bash
watch -n 2 'open logs/training_loss.png'
watch -n 2 'eog logs/training_loss.png'
```

## Output

After training and testing, you'll get:

```
logs/
├── model.pt
├── threshold.npy
└── training_loss.png

anomaly_results.txt
```

Example test output:

```
1. suspicious_doc.txt
   Status: ANOMALY
   Reconstruction Error: 0.045678
   Anomaly Likelihood: 156.23%

2. normal_doc.txt
   Status: Normal
   Reconstruction Error: 0.012345
   Anomaly Likelihood: 42.15%
```
