import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from autoencoder import BertAutoencoder
from utils import DocumentEmbedder, DocumentDataset

import warnings
warnings.filterwarnings('ignore')


def test_model(test_dir, model_path, threshold_path, batch_size, output_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(test_dir):
        raise ValueError(f"Test directory '{test_dir}' not found!")
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model file '{model_path}' not found!")
    
    if not os.path.exists(threshold_path):
        raise ValueError(f"Threshold file '{threshold_path}' not found!")
    
    print("Loading model and threshold...")
    net = BertAutoencoder(input_dim=768, edim=256).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    
    th = np.load(threshold_path)
    print(f"Loaded threshold: {th:.6f}")
    
    print("\nInitializing BERT embedder...")
    x1 = DocumentEmbedder(device=device)
    
    print(f"\nLoading test documents from {test_dir}...")
    test_dataset = DocumentDataset(test_dir, x1)
    dl2 = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    results = []
    
    print("\n" + "="*80)
    print("ANOMALY DETECTION RESULTS")
    print("="*80)
    
    with torch.no_grad():
        for embeddings, filenames in dl2:
            embeddings = embeddings.to(device)
            outputs = net(embeddings)
            
            errors = torch.mean((embeddings - outputs) ** 2, dim=1)
            
            for filename, error in zip(filenames, errors.cpu().numpy()):
                is_anomaly = error > th
                alik = min(100, (error / th) * 100)
                
                results.append({
                    'filename': filename,
                    'reconstruction_error': error,
                    'is_anomaly': is_anomaly,
                    'anomaly_likelihood': alik
                })
    
    results.sort(key=lambda x: x['reconstruction_error'], reverse=True)
    
    for i, result in enumerate(results, 1):
        status = "ANOMALY" if result['is_anomaly'] else "Normal"
        print(f"\n{i}. {result['filename']}")
        print(f"   Status: {status}")
        print(f"   Reconstruction Error: {result['reconstruction_error']:.6f}")
        print(f"   Anomaly Likelihood: {result['anomaly_likelihood']:.2f}%")
    
    print("\n" + "="*80)
    print(f"Summary: {sum(r['is_anomaly'] for r in results)}/{len(results)} documents flagged as anomalous")
    print("="*80)
    
    with open(output_file, 'w') as f:
        f.write("ANOMALY DETECTION RESULTS\n")
        f.write("="*80 + "\n\n")
        for result in results:
            status = "ANOMALY" if result['is_anomaly'] else "Normal"
            f.write(f"File: {result['filename']}\n")
            f.write(f"Status: {status}\n")
            f.write(f"Reconstruction Error: {result['reconstruction_error']:.6f}\n")
            f.write(f"Anomaly Likelihood: {result['anomaly_likelihood']:.2f}%\n\n")
    
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Test autoencoder for anomaly detection')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test documents')
    parser.add_argument('--model_path', type=str, default='logs/model.pt', help='Path to saved model (default: logs/model.pt)')
    parser.add_argument('--threshold_path', type=str, default='logs/threshold.npy', help='Path to saved threshold (default: logs/threshold.npy)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for testing (default: 8)')
    parser.add_argument('--output_file', type=str, default='anomaly_results.txt', help='Output file for results (default: anomaly_results.txt)')
    
    args = parser.parse_args()
    
    test_model(
        test_dir=args.test_dir,
        model_path=args.model_path,
        threshold_path=args.threshold_path,
        batch_size=args.batch_size,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()
