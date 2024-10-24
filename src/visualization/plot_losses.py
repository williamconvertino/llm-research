import os
import matplotlib.pyplot as plt

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../outputs/figures')

def plot_losses(*args, title='Losses', save_name=None, xlabel='Steps', ylabel='Loss', mark_epochs=True, epoch_steps=100):
  
  plt.figure(figsize=(12, 8))
  plt.title = title
  
  max_step = 0
  
  for (label, step_losses, color) in args:
    x, y = zip(*step_losses)
    plt.plot(x, y, label=label, color=color)
    max_step = max(max_step, max(x))
  
  if mark_epochs:
    for x in range(epoch_steps, max_step, epoch_steps):
      plt.axvline(x=x, color='red', linestyle='--', label='Epoch Boundary' if x == epoch_steps else None)
      
  plt.legend()
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  
  if not save_name:
    save_name = title.lower().replace(' ', '_')
  
  if not os.path.exists(base_dir):
    os.makedirs(base_dir, exist_ok=True)
  plt.savefig(os.path.join(base_dir, f'{save_name}.png'))
  
  plt.show()