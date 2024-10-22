import os
from datasets import load_from_disk

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/datasets/tokenized')

def tokenize_dataset(dataset, tokenizer, context_size=512):
  tokenized_dataset_name = f"{dataset.name}_tokenizer=({tokenizer.name})_cs=({context_size})"
  tokenized_dataset_dir = os.path.join(base_dir, tokenized_dataset_name)
  
  if os.path.exists(tokenized_dataset_dir):
    return load_from_disk(tokenized_dataset_dir)
  
  tokenize_dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=context_size), batched=True, remove_columns=['text'])