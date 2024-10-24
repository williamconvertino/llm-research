import os
import matplotlib.pyplot as plt
from src.visualization import plot_losses

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../figures')

def plot_results(train_results, val_results, model):
  
  epoch_train_losses = [(epoch, data['loss']) for (epoch, data) in train_results.items()]
  epoch_val_losses = [(epoch, data['loss']) for (epoch, data) in val_results.items()]
  
  epoch_train_losses = ("Train Loss", epoch_train_losses, 'blue')
  epoch_val_losses = ("Val Loss", epoch_val_losses, 'red')
  
  plot_losses(epoch_train_losses, epoch_val_losses, title=f"Epoch Losses for {model.name}", save_name=f"{model.name}_losses", xlabel='Epochs', ylabel='Loss')
  
  # all_batch_train_losses = []
  # all_val_train_losses = []
  
  # for epoch, data in train_results.items():
  #   for (step, loss) in data['batch_losses']:
      