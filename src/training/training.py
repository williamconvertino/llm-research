import os
import math
import time
import torch
from torch.nn.utils import clip_grad_norm_
from src.visualization import plot_results

model_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models')
results_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../outputs/results')

learning_rate = 1e-5
weight_decay = 0.01
max_grad_norm = 1.0

def step(model, batch):
  input_ids = batch['input_ids'].to(model.device)
  targets = torch.zeros_like(input_ids).to(model.device)
  targets[:, :-1] = input_ids[:, 1:]
  targets[:, -1] = -1
  _, loss = model(input_ids, targets)
  return loss

def train(model, train_dataloader, val_dataloader, num_epochs=10, record_steps=None, v=True, device=None, simulation_name=None):
  
  # Device 
  if device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  device = torch.device(device)
  
  model.to(device)
  model.device = device
  
  # Recording
  if record_steps is None:
    record_steps = len(train_dataloader) // 20
  
  record_steps = min(record_steps, len(train_dataloader))
  record_steps = max(record_steps, 1)
    
  if simulation_name is None:
    simulation_name = model.name

  train_results = {}
  val_results = {}
  
  train_results_path = os.path.join(results_base_dir, f"{simulation_name}_train_results.json")
  val_results_path = os.path.join(results_base_dir, f"{simulation_name}_eval_results.json")
  
  if not os.path.exists(model_base_dir):
    os.makedirs(model_base_dir, exist_ok=True)
  if not os.path.exists(results_base_dir):
    os.makedirs(results_base_dir, exist_ok=True)

  # Training Initialization
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  start_time = time.time()
  print(f"Training model {simulation_name} [Using device: {device}]")
  
  # Training Loop
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
      
      loss = step(model, batch)
      
      epoch_loss += loss.item()
      loss.backward()
      clip_grad_norm_(model.parameters(), max_grad_norm)
      optimizer.step()
      
      if (i + 1) % record_steps == 0 or (i == 0 and epoch == 0):
        
        val_loss = 0
        for v_batch in val_dataloader:
          val_loss += step(model, v_batch).item()
        val_loss /= len(val_dataloader)
          
        train_results[epoch]['batch_losses'].append((i, loss.item()))
        val_results[epoch]['batch_losses'].append((i, val_loss))
        
        most_recent_val_string = f"{val_loss:.4f}"
      
      elapsed_time = time.time() - batch_start_time
      time_remaining = (elapsed_time / (i + 1)) * (len(train_dataloader) - (i + 1))
      time_remaining = time.strftime("%H:%M:%S", time.gmtime(time_remaining))
      print(f"\rEpoch {epoch + 1} | Batch {i + 1} / {len(train_dataloader)} | Train Loss: {loss.item():.4f} | Most Recent Val Loss: {most_recent_val_string} | Batch Time Remaining: {time_remaining}", end='', flush=True)
    
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    
    train_results[epoch]['loss'] = avg_epoch_loss
    val_results[epoch]['loss'] = val_loss
    
    time_elapsed = time.time() - start_time
    time_remaining = (time_elapsed / (epoch + 1)) * (num_epochs - (epoch + 1))
    time_remaining = time.strftime("%H:%M:%S", time.gmtime(time_remaining))
    print(f"Epoch {epoch + 1} / {num_epochs} | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_results[epoch]['loss']:.4f} | Time Remaining: {time_remaining}")
    
    torch.save(model.state_dict(), os.path.join(model_base_dir, f"{simulation_name}_epoch_{epoch}.pt"))
    torch.save(train_results, train_results_path)
    torch.save(val_results, val_results_path)
    
    plot_results(train_results, val_results, model, simulation_name)
    
  print("="*40)
  print(f"Training complete | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_results[epoch]['loss']:.4f}")
  print("="*49)
    
  return train_results, val_results