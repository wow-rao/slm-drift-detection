import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from autoencoder import BertAutoencoder
from utils import DocumentEmbedder, DocumentDataset, TrainingVisualizer

import warnings
warnings.filterwarnings('ignore')


def train_model(train_dir, epochs, batch_size, lr, log_dir, viz_freq):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory '{train_dir}' not found!")
    
    os.makedirs(log_dir, exist_ok=True)
    
    print("Initializing BERT embedder...")
    x1 = DocumentEmbedder(device=device)
    
    print(f"\nLoading training documents from {train_dir}...")
    train_dataset = DocumentDataset(train_dir, x1)
    dl1 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("\nInitializing Autoencoder...")
    net = BertAutoencoder(input_dim=768, edim=256).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    viz = TrainingVisualizer(log_dir=log_dir)
    
    net.train()
    
    print(f"\nTraining for {epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(epochs):
        total_loss = 0
        for embeddings, _ in dl1:
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
    
    net.eval()
    rerr = []
    
    with torch.no_grad():
        for embeddings, _ in dl1:
            embeddings = embeddings.to(device)
            outputs = net(embeddings)
            
            errors = torch.mean((embeddings - outputs) ** 2, dim=1)
            rerr.extend(errors.cpu().numpy())
    
    th = np.percentile(rerr, 95)
    print(f"\nAnomaly threshold (95th percentile): {th:.6f}")
    print(f"Mean training reconstruction error: {np.mean(rerr):.6f}")
    
    model_path = os.path.join(log_dir, 'model.pt')
    threshold_path = os.path.join(log_dir, 'threshold.npy')
    
    torch.save(net.state_dict(), model_path)
    np.save(threshold_path, th)
    
    print(f"\nModel saved to {model_path}")
    print(f"Threshold saved to {threshold_path}")


def main():
    parser = argparse.ArgumentParser(description='Train autoencoder for anomaly detection')
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
