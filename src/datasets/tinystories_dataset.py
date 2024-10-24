import os
import random
import datasets
from torch.utils.data import DataLoader
from src.preprocessing import pad_tokenized_dataset

dataset_name = 'tinystories'
huggingface_path = 'roneneldan/TinyStories'

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets')
raw_dir = os.path.join(base_dir, 'raw')
tokenized_dir = os.path.join(base_dir, 'tokenized')

DEFAULT_TRAIN_SIZE = 2000000
DEFAULT_VAL_SIZE = 19000
DEFAULT_TEST_SIZE = 20000

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
    dataset.save_to_disk(dataset_path)
    
  dataset.name = tokenized_dataset_name
  
  return dataset

def load_tinystories_dataset_preprocessed(tokenizer, context_size=512, use_sliding_window=False, use_random_window=False):
  
  preprocessed_dataset_name = f'{dataset_name}_tokenizer=({tokenizer.name})_cs=({context_size})'
  if use_sliding_window:
    preprocessed_dataset_name += '_sw'
  if use_random_window:
    preprocessed_dataset_name += '_rw'
  
  dataset_path = os.path.join(tokenized_dir, preprocessed_dataset_name)
  if os.path.exists(dataset_path):
    dataset = datasets.load_from_disk(dataset_path)
  else:
    dataset = load_tinystories_dataset_tokenized(tokenizer)
    dataset = pad_tokenized_dataset(dataset, tokenizer, context_size, use_sliding_window, use_random_window)
    dataset.set_format(type='torch', columns=['input_ids'])
    dataset.save_to_disk(dataset_path)
  
  dataset.name = preprocessed_dataset_name
  return dataset
  
def load_tinystories_dataloaders(tokenizer, context_size=512, batch_size=32, train_size=DEFAULT_TEST_SIZE, val_size=DEFAULT_VAL_SIZE, test_size=DEFAULT_TEST_SIZE, use_sw=False, use_rw=False):
  
  dataset = load_tinystories_dataset_preprocessed(tokenizer, context_size, use_sw, use_rw)
  
  assert len(dataset['train']) >= train_size + val_size, f"train_size + val_size ({train_size + val_size}) is greater than the number of training samples ({len(dataset['train'])})"
  assert len(dataset['test']) >= test_size, f"test_size ({test_size}) is greater than the number of test samples ({len(dataset['test'])})"
  
  dataset = datasets.DatasetDict({
    'train': dataset['train'].select(range(train_size)),
    'val': dataset['train'].select(range(train_size, train_size + val_size)),
    'test': dataset['test'].select(range(test_size))
  })
  
  train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
  val_dataloader = DataLoader(dataset['val'], batch_size=batch_size, shuffle=False)
  test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
    
  return train_dataloader, val_dataloader, test_dataloader