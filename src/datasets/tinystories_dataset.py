import os
import random
import datasets

dataset_name = 'tinystories'
huggingface_path = 'roneneldan/TinyStories'

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets')
raw_dir = os.path.join(base_dir, 'raw')
tokenized_dir = os.path.join(base_dir, 'tokenized')

def load_tinystories_dataset_raw():
  if not os.path.exists(raw_dir):
      os.makedirs(raw_dir)
  return datasets.load_dataset(huggingface_path, cache_dir=raw_dir)

def load_tinystories_dataset(train_size=2000000, val_size=20000, test_size=20000):
  
  raw_dataset = load_tinystories_dataset_raw()
  
  assert train_size + val_size <= len(raw_dataset['train']), f"train_size + val_size must be less than or equal to the number of training examples ({len(raw_dataset['train'])})"
  assert test_size <= len(raw_dataset['test']), f"test_size must be less than or equal to the number of test examples ({len(raw_dataset['test'])})" 
  
  dataset = raw_dataset['train'].train_test_split(train_size=train_size, test_size=val_size, shuffle=True, seed=42)
  dataset['test'] = raw_dataset['test'].select(random.sample(range(len(raw_dataset['test'])), test_size))
  
  dataset.name = dataset_name
  
  return dataset