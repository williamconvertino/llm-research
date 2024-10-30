import os
import datasets
from torch.utils.data import DataLoader
from src.preprocessing import apply_sliding_window, apply_padding

DATASET_NAME = 'tinystories'
huggingface_path = 'roneneldan/TinyStories'

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets')

def load_tinystories_dataset_raw():
  dataset_path = os.path.join(BASE_DIR, 'raw', DATASET_NAME)
  if not os.path.exists(dataset_path):
      os.makedirs(dataset_path)
  dataset = datasets.load_dataset(huggingface_path, cache_dir=dataset_path)
  dataset['test'] = dataset.pop('validation')
  
  return dataset

def load_tinystories_dataset_tokenized(tokenizer):
  
  dataset_name = f'{DATASET_NAME}_tokenized_(tokenizer={tokenizer.name})'
  dataset_path = os.path.join(BASE_DIR, 'tokenized', dataset_name)
  
  if os.path.exists(dataset_path):
    dataset = datasets.load_from_disk(dataset_path)
  else:
    dataset = load_tinystories_dataset_raw()
    dataset = dataset.map(lambda x: tokenizer(x['text'], return_attention_mask=False), batched=True)
    print(f"Saving tokenized dataset to {dataset_path}")
    dataset.save_to_disk(dataset_path)
    
  dataset.name = dataset_name
  return dataset

def load_tinystories_dataset_padded(tokenizer, context_size=512):
  
  dataset_name = f'{DATASET_NAME}_padded_(tokenizer={tokenizer.name})_(cs={context_size})'
  dataset_path = os.path.join(BASE_DIR, 'padded', dataset_name)

  if os.path.exists(dataset_path):
    dataset = datasets.load_from_disk(dataset_path)
  else:
    dataset = load_tinystories_dataset_raw()
    dataset.cleanup_cache_files()
    dataset['train'] = dataset['train'].map(lambda x: apply_padding(x, tokenizer, context_size, sw=True), batched=True, remove_columns=['text'])
    dataset['test'] = dataset['test'].map(lambda x: apply_padding(x, tokenizer, context_size), batched=True, remove_columns=['text'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    # dataset.save_to_disk(dataset_path)
    
  dataset.name = dataset_name
  return dataset

def load_tinystories_dataset_sliding_window(tokenizer, context_size=512):
  
  dataset_name = f'{DATASET_NAME}_sw_(tokenizer={tokenizer.name})_(cs={context_size})'
  dataset_path = os.path.join(BASE_DIR, 'sw', dataset_name)
  
  if os.path.exists(dataset_path):
    dataset = datasets.load_from_disk(dataset_path)
  else:
    dataset = load_tinystories_dataset_tokenized(tokenizer)
    dataset.cleanup_cache_files()
    dataset = apply_sliding_window(dataset, tokenizer, context_size)
    print(f"Saving sliding dataset to {dataset_path}")
    dataset.set_format(type='torch', columns=['input_ids'])
    dataset.save_to_disk(dataset_path)
  
  dataset.name = dataset_name
  return dataset
  
def load_tinystories_dataloaders(tokenizer, context_size=512, batch_size=32, train_size=None, val_size=20000, test_size=None, sw=False):
  
  if sw:
    dataset = load_tinystories_dataset_sliding_window(tokenizer, context_size)
  else:
    dataset = load_tinystories_dataset_padded(tokenizer, context_size)
  
  assert len(dataset['train']) >= train_size + val_size, f"train_size + val_size ({train_size + val_size}) is greater than the number of training samples ({len(dataset['train'])})"
  assert len(dataset['test']) >= test_size, f"test_size ({test_size}) is greater than the number of test samples ({len(dataset['test'])})"
  
  if train_size is None:
    train_size = len(dataset['train']) - val_size
    
  if test_size is None:
    test_size = len(dataset['test'])
  
  dataset = datasets.DatasetDict({
    'train': dataset['train'].select(range(train_size)),
    'val': dataset['train'].select(range(train_size, train_size + val_size)),
    'test': dataset['test'].select(range(test_size))
  })
  
  train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
  val_dataloader = DataLoader(dataset['val'], batch_size=batch_size, shuffle=False)
  test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
    
  return train_dataloader, val_dataloader, test_dataloader