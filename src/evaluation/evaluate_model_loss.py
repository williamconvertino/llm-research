import torch

def evaluate_model_loss(model, dataloader):
  model.eval()
  total_loss = 0.0
  for batch in dataloader:
    sequences = batch['input_ids'].to(model.device)
    inputs = sequences[:, :-1]
    targets = sequences[:, 1:].contiguous()
    with torch.no_grad():
      _, loss = model(inputs, targets)
      total_loss += loss.item()
  
  return total_loss / len(dataloader)