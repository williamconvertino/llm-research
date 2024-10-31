import time
import gc
from datasets import DatasetDict, Dataset, IterableDataset
from src.utils.time_remaining import calculate_time_remaining

def apply_sliding_window(dataset, tokenizer, context_size, pct_stride=0.5):
  
  stride = int(context_size * pct_stride)
  
  def split_sliding_window(sequences):
    current_window = []
    print("Splitting windows...")
    start_time = time.time()
    
    for i, sequence in enumerate(sequences):
      current_window.extend(sequence)
      current_window.append(tokenizer.eos_token_id)
      while len(current_window) >= context_size:
        yield current_window[:context_size]
        current_window = current_window[stride:]
      time_remaining = calculate_time_remaining(start_time, i, len(sequences))
      print(f"\r[{i+1}/{len(sequences)}] | Time Remaining: {time_remaining}", end='')  
    print("\nDone splitting windows.")
  
  sw_dataset = DatasetDict({
    'train': IterableDataset.from_generator(lambda: split_sliding_window(dataset['train']['input_ids'])),
    'test': Dataset.from_dict({
      'input_ids': dataset['test']['input_ids'],
      'text': dataset['test']['text']
    })
  })
    
  return sw_dataset