import random

def pad_input_ids(examples, tokenizer, context_size, use_sliding_window=False, use_random_window=False):
  
    assert not (use_sliding_window and use_random_window), "use_sliding_window and use_random_window cannot both be True"
  
    sw_stride = context_size // 2
    new_input_ids = []
        
    def pad_sequence(sequence):
        sequence = sequence[:context_size]
        if len(sequence) < context_size:
            sequence += [tokenizer.pad_token_id] * (context_size - len(sequence))
        return sequence
    
    for input_id in examples['input_ids']:
        if use_sliding_window:
          for i in range(0, len(input_id), sw_stride):
            new_input_ids.append(pad_sequence(input_id[i:i+context_size]))
        elif use_random_window and len(input_id) >= context_size:
          start = random.randint(0, len(input_id) - context_size)
          new_input_ids.append(pad_sequence(input_id[start:start+context_size]))
        else:
          new_input_ids.append(pad_sequence(input_id))

    return {'input_ids': new_input_ids}
  
def pad_tokenized_dataset(dataset, tokenizer, context_size, use_sliding_window=False, use_random_window=False):
  for split in dataset:
    use_sw = use_sliding_window and split == 'train'
    use_rw = use_random_window and split == 'train'
    remove_columns = ['input_ids', 'text'] if use_sliding_window or use_random_window else ['input_ids']
    dataset[split] = dataset[split].map(lambda x: pad_input_ids(x, tokenizer, context_size, use_sw, use_rw), batched=True, remove_columns=remove_columns)
  return dataset