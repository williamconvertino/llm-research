import os
import time
import datasets
from datasets import Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader

HF_PATH = 'roneneldan/TinyStories'
DATASET_NAME = 'tinystories'
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets/', DATASET_NAME)
RAW_DIR = f'{BASE_DIR}/raw'

def load_tinystories_dataset():
  return datasets.load_dataset(HF_PATH, cache_dir=RAW_DIR)

def load_tinystories_dataset_padded(tokenizer, context_size):

  dataset_name = f'{DATASET_NAME}_preprocessed_(cs={context_size})'
  dataset_path = f'{BASE_DIR}/preprocessed/{dataset_name}'
  
  if os.path.exists(dataset_path):
    print(f"Loading dataset from {dataset_path}")
    return datasets.load_from_disk(dataset_path)
    
  dataset = load_tinystories_dataset()
  
  dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=context_size), batched=True)
  dataset.save_to_disk(dataset_path)
  
  return dataset
  
def load_tinystories_dataset_sliding_window(tokenizer, context_size):
  
  dataset_name = f'{DATASET_NAME}_sliding_window_(cs={context_size})'
  dataset_path = f'{BASE_DIR}/sliding_window/{dataset_name}'
  
  if os.path.exists(dataset_path):
    print(f"Loading dataset from {dataset_path}")
    return datasets.load_from_disk(dataset_path)
  
  dataset = datasets.load_dataset(HF_PATH, cache_dir=RAW_DIR, streaming=True)
  
  def generate_sliding_windows(split):
    stride = context_size // 2
    input_ids = []
    current_window = []
    
    start_time = time.time()
    dataset_size = 2100000
    print("Generating windows...")
    
    for i, sequence in enumerate(split):
      sequence = tokenizer.encode(sequence['text'])
      current_window.extend(sequence)
      current_window.append(tokenizer.eos_token_id)
      while len(current_window) >= context_size:
        input_ids.append(current_window[:context_size])
        current_window = current_window[stride:]
      elapsed_time = time.time() - start_time
      time_remaining = elapsed_time * (dataset_size / (i + 1) - 1)
      time_remaining = time.strftime("%H:%M:%S", time.gmtime(time_remaining))
      print(f"\r[{i+1}/{dataset_size}] | Time remaining: {time_remaining}", end='')
    
    print("\nDone generating windows.")  
    return input_ids
  
  dataset = DatasetDict({
    'train': Dataset.from_dict({
      'input_ids': generate_sliding_windows(dataset['train'])
    }),
    'validation': Dataset.from_dict({
      'input_ids': generate_sliding_windows(dataset['validation'])
    })
  })

  print(f"Saving dataset to {dataset_path}")
  dataset.save_to_disk(dataset_path)
  
  return dataset
  
def load_tinystories_dataloaders(tokenizer, context_size=512, batch_size=32):
  
  # dataset = load_tinystories_dataset_sliding_window(tokenizer, context_size)
  dataset = load_tinystories_dataset_padded(tokenizer, context_size)
  
  def collate_fn(examples):
    input_ids = torch.stack([torch.tensor(example['input_ids']) for example in examples])
    
    if 'attention_mask' in examples[0]:
      attention_mask = torch.stack([torch.tensor(example['attention_mask']) for example in examples])
    else:
      attention_mask = None
      
    return {
      'input_ids': input_ids,
      'attention_mask': attention_mask
    }
  
  train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
  val_dataloader = DataLoader(dataset['validation'], batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
  return train_dataloader, val_dataloader