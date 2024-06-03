import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

class GaLore(nn.Module):
    def __init__(self, model, rank, alpha, optimizer, subspace_freq):
        super().__init__()
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.optimizer = optimizer
        self.subspace_freq = subspace_freq
        self.step = 0
        
        # Initialize projection matrices using SVD
        self.projections = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                U, S, V = torch.svd(param.data)
                self.projections[name] = (U[:, :self.rank], V[:, :self.rank])
    
    def update_projections(self):
        # Update projection matrices using SVD every subspace_freq steps
        if self.step % self.subspace_freq == 0:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    U, S, V = torch.svd(param.data)
                    self.projections[name] = (U[:, :self.rank], V[:, :self.rank])
    
    def project(self, grad, name):
        # Project gradient to low-rank subspace
        U, V = self.projections[name]
        return U @ (V.t() @ grad)
    
    def project_back(self, grad, name):
        # Project gradient back to original space
        U, V = self.projections[name]
        return U @ grad @ V.t()
    
    def step(self):
        # Perform one optimization step
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad = param.grad
                lr_grad = self.project(grad, name)
                lr_update = self.optimizer.step(lr_grad)
                update = self.project_back(lr_update, name)
                param.data += self.alpha * update
        
        self.update_projections()
        self.step += 1

def train(model, dataloader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    galore = GaLore(model, rank=256, alpha=0.01, optimizer=optimizer, subspace_freq=100)
    
    for epoch in range(epochs):
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            galore.step()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Load pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Prepare dataset
dataset = ...  # Load your dataset
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train the model
epochs = 10
train(model, dataloader, epochs)