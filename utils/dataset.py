from torch.utils.data import Dataset
from pathlib import Path


class DocumentDataset(Dataset):
    
    def __init__(self, folder_path, x1):
        self.folder_path = Path(folder_path)
        self.x1 = x1
        self.file_paths = sorted(list(self.folder_path.glob('*.txt')))
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No .txt files found in {folder_path}")
        
        print(f"Found {len(self.file_paths)} documents in {folder_path}")
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        embedding = self.x1.get_embedding(text)
        return embedding, str(file_path.name)
