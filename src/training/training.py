import os
import torch
import transformers
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from src.evaluation import evaluate_model_loss

model_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models')

learning_rate = 0.0001
weight_decay = 0.01
max_grad_norm = 1.0
p_warmup_steps = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def train(model, train_dataloader, val_dataloader, num_epochs=10, record_steps=5, v=True):
    
  num_warmup_steps = p_warmup_steps * len(train_dataloader)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, len(train_dataloader))
  
  scaler = GradScaler(device)
  
  if v:
    print("="*40)
    print(f"Training model {model.name} [Using device: {device}]")
    print("="*40)
  
  model.to(device)
  model.device = device
  
  train_results = {}
  eval_results = {}
  
  for epoch in range(num_epochs):
    
    model.train()
    
    train_results[epoch] = {
      'batch_losses': []
    }
    
    eval_results[epoch] = {
      'batch_losses': []
    }
    
    epoch_loss = 0
    
    for i, batch in enumerate(train_dataloader):
      
      optimizer.zero_grad()
      
      with autocast(device.type):
        sequences = batch['input_ids'].to(device)
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:].contiguous()
          
        _, loss = model(inputs, targets)
        
        epoch_loss += loss.item()
        if (i + 1) % record_steps == 0 or (i == 0 and epoch == 0):
          train_results[epoch]['batch_losses'].append((i, loss.item()))
          eval_results[epoch]['batch_losses'].append((i, evaluate_model_loss(model, val_dataloader)))
        
      scaler.scale(loss).backward()
      clip_grad_norm_(model.parameters(), max_grad_norm)
      scaler.step(optimizer)
      scaler.update()
      scheduler.step()
      
      if v:
        print(f"Epoch {epoch + 1} | Batch {i + 1} / {len(train_dataloader)} | Train Loss: {loss.item()}", end='\r')
    
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    
    train_results[epoch]['loss'] = avg_epoch_loss
    eval_results[epoch]['loss'] = evaluate_model_loss(model, val_dataloader)
    
    if v:
      print(f"Epoch {epoch + 1} / {num_epochs} | Train Loss: {avg_epoch_loss} | Val Loss: {eval_results[epoch]['loss']}")
      
  if v:
    print("="*40)
    print(f"Training complete | Train Loss: {avg_epoch_loss} | Val Loss: {eval_results[epoch]['loss']}")
    print("="*49)
    
  return train_results, eval_results