import random

def apply_padding(examples, tokenizer, context_size, sw=False):
  
  input_ids = []
  attention_mask = []
  
  stride = context_size // 2
    
  input_ids = []
  attention_mask = []
  
  for example in examples['text']:
    if sw:
      chunks = tokenizer(
        example,
        padding='max_length',
        max_length=context_size,
        return_overflowing_tokens=True,
        stride=stride,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
      )
      input_ids.extend(chunks['input_ids'])
      attention_mask.extend(chunks['attention_mask'])
    else:
      chunks = tokenizer(
        example,
        padding='max_length',
        max_length=context_size,
        return_overflowing_tokens=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
      )
      input_ids.append(chunks['input_ids'])
      attention_mask.append(chunks['attention_mask'])
  
  return { 'input_ids': input_ids, 'attention_mask': attention_mask }