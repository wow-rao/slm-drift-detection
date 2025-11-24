# Document Anomaly Detection - V2 (Chunk-Based Training)

## What's Different in V2?

### V1 Approach (Average-Based):
1. Document → Chunks → BERT embeddings
2. **Average all chunk embeddings** → Single vector per document
3. Train autoencoder on averaged embeddings
4. Anomaly score = reconstruction error of averaged embedding

### V2 Approach (Chunk-Based):
1. Document → Chunks → BERT embeddings
2. **Train on individual chunk embeddings** (each chunk is a separate training sample)
3. For testing:
   - Get all chunk embeddings for document
   - Calculate reconstruction loss for each chunk
   - **Sum and normalize by number of chunks**
4. Anomaly score = normalized sum of chunk reconstruction losses

## Key Benefits of V2

- More granular training on document segments
- Captures chunk-level patterns better
- Normalized scores account for variable document lengths
- Can potentially detect localized anomalies within documents

## Quick Start

### Training

```bash
python src/exec/v2/train.py --epochs 50 --train_dir data/train
```

### Testing

```bash
python src/exec/v2/play.py --test_dir data/test
```

## Training Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--epochs` | Yes | - | Number of training epochs |
| `--train_dir` | Yes | - | Directory with training documents |
| `--batch_size` | No | 8 | Batch size for training |
| `--lr` | No | 0.001 | Learning rate |
| `--log_dir` | No | logs | Directory to save model/logs |
| `--viz_freq` | No | 5 | Update visualization every N epochs |

## Testing Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--test_dir` | Yes | - | Directory with test documents |
| `--model_path` | No | logs/model.pt | Path to trained model |
| `--threshold_path` | No | logs/threshold.npy | Path to threshold file |
| `--output_file` | No | anomaly_results_v2.txt | Output file for results |

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

## How Training Works

1. **Document Processing**: Each document is chunked into segments that fit BERT's token limit
2. **Chunk Extraction**: All chunks from all documents are extracted (e.g., 10 documents × 100 chunks = 1000 training samples)
3. **Individual Training**: Autoencoder trains on each chunk embedding independently
4. **Threshold Calculation**: 
   - Process each training document
   - Calculate reconstruction loss for each chunk
   - Sum and normalize by number of chunks per document
   - Set threshold at 95th percentile of normalized errors

## How Testing Works

1. **Load Model**: Load trained autoencoder and threshold
2. **Process Test Documents**: For each document:
   - Extract all chunk embeddings
   - Pass each chunk through autoencoder
   - Calculate reconstruction loss for each chunk
   - Sum losses and divide by number of chunks (normalization)
3. **Anomaly Detection**: Compare normalized loss against threshold
4. **Output Results**: Print and save results with anomaly likelihood scores

## Output Format

```
================================================================================
ANOMALY DETECTION RESULTS
================================================================================

1. suspicious_doc.txt
   Status: ANOMALY
   Reconstruction Error (normalized): 0.045678
   Number of Chunks: 127
   Anomaly Likelihood: 156.23%

2. normal_doc.txt
   Status: Normal
   Reconstruction Error (normalized): 0.012345
   Number of Chunks: 98
   Anomaly Likelihood: 42.15%

================================================================================
Summary: 1/2 documents flagged as anomalous
================================================================================
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

## Complete Workflow

```bash
# 1. Prepare data
mkdir -p data/train data/test

# 2. Train model
python src/exec/v2/train.py --epochs 50 --train_dir data/train

# 3. Test for anomalies
python src/exec/v2/play.py --test_dir data/test

# 4. Review results
cat anomaly_results_v2.txt
```

## Differences from V1

| Aspect | V1 | V2 |
|--------|----|----|
| Training unit | Document (averaged chunks) | Individual chunks |
| Training samples | N documents | N documents × M chunks |
| Anomaly score | Single reconstruction error | Sum of chunk errors / num_chunks |
| Normalization | None | By number of chunks |
| Output file | anomaly_results.txt | anomaly_results_v2.txt |

## When to Use V2 vs V1

**Use V2 when:**
- You want more granular, chunk-level learning
- Documents have highly variable lengths
- You suspect anomalies may be localized to specific sections
- You have many documents to train on

**Use V1 when:**
- You want document-level holistic representation
- Training time is a concern (V1 is faster)
- You have fewer training documents
- Documents are relatively uniform in length

## Troubleshooting

**Training takes longer than V1:**
- This is expected as V2 trains on many more samples (all chunks vs documents)
- Reduce `--batch_size` or increase GPU memory

**Different results from V1:**
- V2 and V1 use fundamentally different approaches
- Both are valid; compare which works better for your use case

**Memory issues:**
- ChunkDataset loads all chunks into memory
- For very large datasets, consider processing in batches or using V1
