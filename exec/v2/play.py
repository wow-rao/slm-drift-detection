import torch
import numpy as np
import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from autoencoder import BertAutoencoder
from utils import DocumentEmbedder

import warnings
warnings.filterwarnings('ignore')


def test_model(test_dir, model_path, threshold_path, output_file):
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
    folder_path = Path(test_dir)
    file_paths = sorted(list(folder_path.glob('*.txt')))
    
    if len(file_paths) == 0:
        raise ValueError(f"No .txt files found in {test_dir}")
    
    print(f"Found {len(file_paths)} documents")
    
    results = []
    
    print("\n" + "="*80)
    print("ANOMALY DETECTION RESULTS")
    print("="*80)
    
    with torch.no_grad():
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            chunks = x1.chunk_text(text)
            
            if len(chunks) == 0:
                print(f"\nWarning: {file_path.name} has no valid chunks, skipping...")
                continue
            
            chunk_errors = []
            for chunk in chunks:
                input_ids = torch.tensor([[x1.tokenizer.cls_token_id] + 
                                         chunk + 
                                         [x1.tokenizer.sep_token_id]]).to(device)
                
                outputs = x1.model(input_ids)
                cemb = outputs.last_hidden_state[:, 0, :].squeeze()
                
                reconstructed = net(cemb.unsqueeze(0))
                error = torch.mean((cemb - reconstructed.squeeze()) ** 2).item()
                chunk_errors.append(error)
            
            normalized_error = sum(chunk_errors) / len(chunk_errors)
            
            is_anomaly = normalized_error > th
            alik = (normalized_error / th) * 100
            
            results.append({
                'filename': str(file_path.name),
                'reconstruction_error': normalized_error,
                'num_chunks': len(chunks),
                'is_anomaly': is_anomaly,
                'anomaly_likelihood': alik
            })
    
    results.sort(key=lambda x: x['reconstruction_error'], reverse=True)
    
    for i, result in enumerate(results, 1):
        status = "ANOMALY" if result['is_anomaly'] else "Normal"
        print(f"\n{i}. {result['filename']}")
        print(f"   Status: {status}")
        print(f"   Reconstruction Error (normalized): {result['reconstruction_error']:.6f}")
        print(f"   Number of Chunks: {result['num_chunks']}")
        print(f"   Anomaly Likelihood: {result['anomaly_likelihood']:.2f}%")
    
    print("\n" + "="*80)
    print(f"Summary: {sum(r['is_anomaly'] for r in results)}/{len(results)} documents flagged as anomalous")
    print("="*80)
    
    with open(output_file, 'w') as f:
        f.write("ANOMALY DETECTION RESULTS (v2 - Chunk-based)\n")
        f.write("="*80 + "\n\n")
        for result in results:
            status = "ANOMALY" if result['is_anomaly'] else "Normal"
            f.write(f"File: {result['filename']}\n")
            f.write(f"Status: {status}\n")
            f.write(f"Reconstruction Error (normalized): {result['reconstruction_error']:.6f}\n")
            f.write(f"Number of Chunks: {result['num_chunks']}\n")
            f.write(f"Anomaly Likelihood: {result['anomaly_likelihood']:.2f}%\n\n")
    
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Test autoencoder for anomaly detection (v2 - chunk-based)')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test documents')
    parser.add_argument('--model_path', type=str, default='logs/model.pt', help='Path to saved model (default: logs/model.pt)')
    parser.add_argument('--threshold_path', type=str, default='logs/threshold.npy', help='Path to saved threshold (default: logs/threshold.npy)')
    parser.add_argument('--output_file', type=str, default='anomaly_results_v2.txt', help='Output file for results (default: anomaly_results_v2.txt)')
    
    args = parser.parse_args()
    
    test_model(
        test_dir=args.test_dir,
        model_path=args.model_path,
        threshold_path=args.threshold_path,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()
