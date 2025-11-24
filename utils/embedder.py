import torch
from transformers import BertTokenizer, BertModel


class DocumentEmbedder:
    
    def __init__(self, model_name='bert-base-uncased', max_length=512, chunk_overlap=50, device='cuda'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.max_length = max_length
        self.chunk_overlap = chunk_overlap
        self.device = device
        
    def chunk_text(self, text, max_tokens=450):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        cks = []
        
        step_size = max_tokens - self.chunk_overlap
        for i in range(0, len(tokens), step_size):
            chunk_tokens = tokens[i:i + max_tokens]
            if len(chunk_tokens) > 0:
                cks.append(chunk_tokens)
                
        return cks
    
    def get_embedding(self, text):
        cks = self.chunk_text(text)
        
        if len(cks) == 0:
            return torch.zeros(768).to(self.device)
        
        embeddings = []
        
        with torch.no_grad():
            for chunk in cks:
                input_ids = torch.tensor([[self.tokenizer.cls_token_id] + 
                                         chunk + 
                                         [self.tokenizer.sep_token_id]]).to(self.device)
                
                outputs = self.model(input_ids)
                cemb = outputs.last_hidden_state[:, 0, :].squeeze()
                embeddings.append(cemb)
        
        demb = torch.stack(embeddings).mean(dim=0)
        return demb
