import os
import time
import datasets
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

HF_PATH = 'roneneldan/TinyStories'
DATASET_NAME = 'tinystories'
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets/', DATASET_NAME)
RAW_DIR = f'{BASE_DIR}/raw'

VAL_SIZE = 20000

def load_tinystories_dataset():
  return datasets.load_dataset(HF_PATH, cache_dir=RAW_DIR)

def load_tinystories_dataset_preprocessed(tokenizer, context_size):
  
  dataset_name = f'{DATASET_NAME}_preprocessed_(cs={context_size})'
  dataset_path = f'{BASE_DIR}/preprocessed/{dataset_name}'
  
  if os.path.exists(dataset_path):
    print(f"Loading dataset from {dataset_path}")
    return datasets.load_from_disk(dataset_path)
  
  train_dataset = datasets.load_dataset(HF_PATH, cache_dir=RAW_DIR, split=f'train', streaming=True)
  test_dataset = datasets.load_dataset(HF_PATH, cache_dir=RAW_DIR, split='validation')
  
  stride = context_size // 2
  input_ids = []
  current_window = []
  
  start_time = time.time()
  dataset_size = 2100000
  print("Generating windows...")
  
  for i, sequence in enumerate(train_dataset):
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
  
  train_dataset = Dataset.from_dict({'input_ids': input_ids})
  
  tv_split = train_dataset.train_test_split(test_size=VAL_SIZE)
  train_dataset = tv_split['train']
  val_dataset = tv_split['test']
  
  test_dataset = test_dataset.map(lambda x: {'input_ids': tokenizer.encode(x['text'][0])[:context_size]})
  
  dataset = DatasetDict({
    'train': train_dataset,
    'val': val_dataset,
    'test': test_dataset
  })
  
  dataset.set_format(type='torch', columns=['input_ids'])
  dataset.save_to_disk(dataset_path)
  
  return dataset
  
def load_tinystories_dataloaders(tokenizer, context_size=512, batch_size=32):
  
  dataset = load_tinystories_dataset_preprocessed(tokenizer, context_size)
  
  train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
  val_dataloader = DataLoader(dataset['val'], batch_size=batch_size, shuffle=False)
  test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
    
  return train_dataloader, val_dataloader, test_dataloader