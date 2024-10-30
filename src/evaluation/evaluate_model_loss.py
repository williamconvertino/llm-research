import torch

def evaluate_model_loss(model, tokenizer, dataloader):
  
  model.eval()
  total_loss = 0.0
  
  for batch in dataloader:
    
    input_ids = batch['input_ids'].to(model.device)
    
    if batch['attention_mask'] is not None:
      attention_mask = batch['attention_mask'].to(model.device)
    else:
      attention_mask = None
      
    targets = torch.zeros_like(input_ids).to(model.device)
    targets[:, :-1] = input_ids[:, 1:]
    targets[:, -1] = tokenizer.pad_token_id
    
    with torch.no_grad():
      _, loss = model(input_ids, targets, attention_mask=attention_mask, padding_token=tokenizer.pad_token_id)
      total_loss += loss.item()
  
  return total_loss / len(dataloader)