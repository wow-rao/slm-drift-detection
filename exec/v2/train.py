import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from autoencoder import BertAutoencoder
from utils import DocumentEmbedder, TrainingVisualizer

import warnings
warnings.filterwarnings('ignore')


class ChunkDataset(Dataset):
    
    def __init__(self, folder_path, x1):
        self.folder_path = Path(folder_path)
        self.x1 = x1
        self.file_paths = sorted(list(self.folder_path.glob('*.txt')))
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No .txt files found in {folder_path}")
        
        print(f"Found {len(self.file_paths)} documents in {folder_path}")
        
        self.chunk_embeddings = []
        self.doc_info = []
        
        print("Processing documents and extracting chunks...")
        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            chunks = self.x1.chunk_text(text)
            
            if len(chunks) == 0:
                continue
            
            with torch.no_grad():
                for chunk in chunks:
                    input_ids = torch.tensor([[self.x1.tokenizer.cls_token_id] + 
                                             chunk + 
                                             [self.x1.tokenizer.sep_token_id]]).to(self.x1.device)
                    
                    outputs = self.x1.model(input_ids)
                    cemb = outputs.last_hidden_state[:, 0, :].squeeze()
                    self.chunk_embeddings.append(cemb.cpu())
            
            self.doc_info.append({
                'filename': str(file_path.name),
                'num_chunks': len(chunks)
            })
        
        print(f"Extracted {len(self.chunk_embeddings)} total chunks from {len(self.doc_info)} documents")
        
    def __len__(self):
        return len(self.chunk_embeddings)
    
    def __getitem__(self, idx):
        return self.chunk_embeddings[idx]


def train_model(train_dir, epochs, batch_size, lr, log_dir, viz_freq):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory '{train_dir}' not found!")
    
    os.makedirs(log_dir, exist_ok=True)
    
    print("Initializing BERT embedder...")
    x1 = DocumentEmbedder(device=device)
    
    print(f"\nLoading training documents from {train_dir}...")
    train_dataset = ChunkDataset(train_dir, x1)
    dl1 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("\nInitializing Autoencoder...")
    net = BertAutoencoder(input_dim=768, edim=256).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    viz = TrainingVisualizer(log_dir=log_dir)
    
    net.train()
    
    print(f"\nTraining for {epochs} epochs on individual chunks...")
    print("=" * 80)
    
    for epoch in range(epochs):
        total_loss = 0
        for embeddings in dl1:
            embeddings = embeddings.to(device)
            
            outputs = net(embeddings)
            loss = criterion(outputs, embeddings)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dl1)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        if (epoch + 1) % viz_freq == 0:
            viz.update(epoch + 1, avg_loss)
    
    print("=" * 80)
    print("Training completed!")
    
    viz.save_final()
    
    print("\nCalculating document-level reconstruction errors for threshold...")
    
    x1_test = DocumentEmbedder(device=device)
    folder_path = Path(train_dir)
    file_paths = sorted(list(folder_path.glob('*.txt')))
    
    net.eval()
    doc_errors = []
    
    with torch.no_grad():
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            chunks = x1_test.chunk_text(text)
            
            if len(chunks) == 0:
                continue
            
            chunk_errors = []
            for chunk in chunks:
                input_ids = torch.tensor([[x1_test.tokenizer.cls_token_id] + 
                                         chunk + 
                                         [x1_test.tokenizer.sep_token_id]]).to(device)
                
                outputs_bert = x1_test.model(input_ids)
                cemb = outputs_bert.last_hidden_state[:, 0, :].squeeze()
                
                reconstructed = net(cemb.unsqueeze(0))
                error = torch.mean((cemb - reconstructed.squeeze()) ** 2).item()
                chunk_errors.append(error)
            
            normalized_error = sum(chunk_errors) / len(chunk_errors)
            doc_errors.append(normalized_error)
    
    th = np.percentile(doc_errors, 95)
    print(f"\nAnomaly threshold (95th percentile): {th:.6f}")
    print(f"Mean training document reconstruction error: {np.mean(doc_errors):.6f}")
    
    model_path = os.path.join(log_dir, 'model.pt')
    threshold_path = os.path.join(log_dir, 'threshold.npy')
    
    torch.save(net.state_dict(), model_path)
    np.save(threshold_path, th)
    
    print(f"\nModel saved to {model_path}")
    print(f"Threshold saved to {threshold_path}")


def main():
    parser = argparse.ArgumentParser(description='Train autoencoder for anomaly detection (v2 - chunk-based)')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--train_dir', type=str, required=True, help='Directory containing training documents')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training (default: 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save model and logs (default: logs)')
    parser.add_argument('--viz_freq', type=int, default=5, help='Frequency to update visualization (default: 5)')
    
    args = parser.parse_args()
    
    train_model(
        train_dir=args.train_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_dir=args.log_dir,
        viz_freq=args.viz_freq
    )


if __name__ == "__main__":
    main()
