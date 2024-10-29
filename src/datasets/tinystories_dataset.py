import os
import datasets
from torch.utils.data import DataLoader
from src.preprocessing import apply_sliding_window

dataset_name = 'tinystories'
huggingface_path = 'roneneldan/TinyStories'

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets')
raw_dir = os.path.join(base_dir, 'raw')
tokenized_dir = os.path.join(base_dir, 'tokenized')
preprocessed_dir = os.path.join(base_dir, 'preprocessed')

DEFAULT_VAL_SIZE = 20000

def load_tinystories_dataset_raw():
  if not os.path.exists(raw_dir):
      os.makedirs(raw_dir)
  return datasets.load_dataset(huggingface_path, cache_dir=raw_dir)

def load_tinystories_dataset_tokenized(tokenizer):

  tokenized_dataset_name = f'{dataset_name}_tokenizer=({tokenizer.name})'
  dataset_path = os.path.join(tokenized_dir, tokenized_dataset_name)
  
  if os.path.exists(dataset_path):
    dataset = datasets.load_from_disk(dataset_path)
  else:
    dataset = load_tinystories_dataset_raw()

    dataset = datasets.DatasetDict({
      'train': dataset['train'],
      'test': dataset['validation'],
    })
    
    dataset = dataset.map(lambda x: tokenizer(x['text'], return_attention_mask=False), batched=True)
    print(f"Saving tokenized dataset to {dataset_path}")
    dataset.save_to_disk(dataset_path)
    
  dataset.name = tokenized_dataset_name
  
  return dataset

def load_tinystories_dataset_preprocessed(tokenizer, context_size=512):
  
  preprocessed_dataset_name = f'{dataset_name}_preprocessed_tokenizer=({tokenizer.name})_cs=({context_size})'
  dataset_path = os.path.join(preprocessed_dir, preprocessed_dataset_name)
  
  if os.path.exists(dataset_path):
    dataset = datasets.load_from_disk(dataset_path)
  else:
    dataset = load_tinystories_dataset_tokenized(tokenizer)
    dataset = apply_sliding_window(dataset, tokenizer, context_size)
    print(f"Saving preprocessed dataset to {dataset_path}")
    dataset.set_format(type='torch', columns=['input_ids'])
    dataset.save_to_disk(dataset_path)
  
  dataset.name = preprocessed_dataset_name
  
  return dataset
  
def load_tinystories_dataloaders(tokenizer, context_size=512, batch_size=32, train_size=None, val_size=DEFAULT_VAL_SIZE, test_size=None):
  
  dataset = load_tinystories_dataset_preprocessed(tokenizer, context_size)
  
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