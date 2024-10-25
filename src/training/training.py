import os
import time
import torch
import transformers
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from src.evaluation import evaluate_model_loss

model_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models')
results_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../outputs/results')

learning_rate = 0.0001
weight_decay = 0.01
max_grad_norm = 1.0
p_warmup_steps = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_dataloader, val_dataloader, num_epochs=10, record_steps=25000, v=True):
    
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
  val_results = {}
  
  train_results_path = os.path.join(results_base_dir, f"{model.name}_train_results.json")
  val_results_path = os.path.join(results_base_dir, f"{model.name}_eval_results.json")
  
  if not os.path.exists(model_base_dir):
    os.makedirs(model_base_dir, exist_ok=True)
  if not os.path.exists(results_base_dir):
    os.makedirs(results_base_dir, exist_ok=True)
  if os.path.exists(train_results_path):
    train_results = torch.load(train_results_path)
  
  start_time = time.time()
  
  for epoch in range(num_epochs):
    
    model.train()
    
    train_results[epoch] = {
      'batch_losses': []
    }
    
    val_results[epoch] = {
      'batch_losses': []
    }
    
    epoch_loss = 0
    batch_start_time = time.time()
    
    most_recent_val_string = "N/A"
    
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
          val_results[epoch]['batch_losses'].append((i, evaluate_model_loss(model, val_dataloader)))
          most_recent_val_string = f"{val_results[epoch]['batch_losses'][-1][1]:.4f}"

      scaler.scale(loss).backward()
      clip_grad_norm_(model.parameters(), max_grad_norm)
      scaler.step(optimizer)
      scaler.update()
      scheduler.step()
      
      if v:
        elapsed_time = time.time() - batch_start_time
        time_remaining = (elapsed_time / (i + 1)) * (len(train_dataloader) - (i + 1))
        time_remaining = time.strftime("%H:%M:%S", time.gmtime(time_remaining))
        print(f"\rEpoch {epoch + 1} | Batch {i + 1} / {len(train_dataloader)} | Train Loss: {loss.item():.4f} | Most Recent Val Loss: {most_recent_val_string} | Batch Time Remaining: {time_remaining}", end='', flush=True)
    
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    
    train_results[epoch]['loss'] = avg_epoch_loss
    val_results[epoch]['loss'] = evaluate_model_loss(model, val_dataloader)
    
    if v:
      time_elapsed = time.time() - start_time
      time_remaining = (time_elapsed / (epoch + 1)) * (num_epochs - (epoch + 1))
      time_remaining = time.strftime("%H:%M:%S", time.gmtime(time_remaining))
      print(f"Epoch {epoch + 1} / {num_epochs} | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_results[epoch]['loss']:.4f} | Time Remaining: {time_remaining}")
      
    torch.save(model.state_dict(), os.path.join(model_base_dir, f"{model.name}_epoch_{epoch}.pt"))
    torch.save(train_results, train_results_path)
    torch.save(val_results, val_results_path)
    
  if v:
    print("="*40)
    print(f"Training complete | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_results[epoch]['loss']:.4f}")
    print("="*49)
    
  return train_results, val_results