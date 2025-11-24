import matplotlib.pyplot as plt
import os


class TrainingVisualizer:
    
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.losses = []
        self.epochs = []
        os.makedirs(log_dir, exist_ok=True)
        
    def update(self, epoch, loss):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self._plot()
        
    def _plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.losses, 'b-', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Over Time', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_loss.png'), dpi=150)
        plt.close()
        
    def save_final(self):
        if len(self.losses) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(self.epochs, self.losses, 'b-', linewidth=2, label='Training Loss')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Final Training Loss', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'final_training_loss.png'), dpi=150)
            plt.close()
            print(f"Final loss plot saved to {self.log_dir}/final_training_loss.png")
