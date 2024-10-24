import os
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from tokenizers import ByteLevelBPETokenizer
from src.datasets.tinystories_dataset import load_tinystories_dataset_raw

base_name = 'tinystories_tokenizer'
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/tokenizers')

def load_tinystories_tokenizer(vocab_size=10000):
  
  tokenizer_name = f'{base_name}_{vocab_size // 1000}k'
  tokenizer_dir = os.path.join(base_dir, tokenizer_name)
  vocab_dir = os.path.join(tokenizer_dir, 'vocab.json')
  merges_dir = os.path.join(tokenizer_dir, 'merges.txt')
  
  if not os.path.exists(vocab_dir) or not os.path.exists(merges_dir):
    print(f"Training tokenizer {tokenizer_name} with vocab_size={vocab_size}")
    tokenizer = ByteLevelBPETokenizer() # Note that we only use the ByteLevelBPETokenizer for training (the final tokenizer will be a GPT2TokenizerFast)
    dataset = load_tinystories_dataset_raw()
    tokenizer.train_from_iterator(dataset['train']['text'], vocab_size=vocab_size-1, min_frequency=5) # We set vocab_size-1 because we will later add a padding token
    del dataset
    
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_model(tokenizer_dir)
  
  tokenizer = GPT2TokenizerFast(vocab_file=vocab_dir, merges_file=merges_dir)
  tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  tokenizer.name = tokenizer_name
  
  return tokenizer