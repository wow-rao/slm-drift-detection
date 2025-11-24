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
