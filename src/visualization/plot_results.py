import os
import matplotlib.pyplot as plt
from src.visualization import plot_losses

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../figures')

def plot_results(train_results, val_results, model, simulation_name=None):
  
  if simulation_name is None:
    simulation_name = model.name
  
  epoch_train_losses = [(epoch, data['loss']) for (epoch, data) in train_results.items()]
  epoch_val_losses = [(epoch, data['loss']) for (epoch, data) in val_results.items()]
  
  epoch_train_losses = ("Train Loss", epoch_train_losses, 'blue')
  epoch_val_losses = ("Val Loss", epoch_val_losses, 'red')
  
  plot_losses(epoch_train_losses, epoch_val_losses, title=f"Epoch Losses for {simulation_name}", save_name=f"{simulation_name}_losses", xlabel='Epochs', ylabel='Loss')
  
  epoch_steps = None
  
  all_train_losses = []
  all_val_losses = []
  
  for i, ((train_epoch, train_data), (val_epoch, val_data)) in enumerate(zip(train_results.items(), val_results.items())):
    train_batch_losses = train_data['batch_losses']
    val_batch_losses = val_data['batch_losses']
    
    if epoch_steps is None:
      epoch_steps = train_batch_losses[-1][0]
    
    all_train_losses.extend([(step + (i * epoch_steps), loss) for (step, loss) in train_batch_losses])
    all_val_losses.extend([(step + (i * epoch_steps), loss) for (step, loss) in val_batch_losses])
    
    train_batch_losses = (f"Train Losses for Epoch {train_epoch + 1}", train_batch_losses, 'blue')
    val_batch_losses = (f"Val Losses for Epoch {val_epoch + 1}", val_batch_losses, 'red')
    
    plot_losses(train_batch_losses, val_batch_losses, title=f"Batch Losses for {model.name} at Epoch {train_epoch + 1}", save_name=f"{simulation_name}_epoch_{train_epoch + 1}_losses", xlabel='Step', ylabel='Loss')
    
    
  plot_losses(("Train Loss", all_train_losses, 'blue'), ("Val Loss", all_val_losses, 'red'), title=f"All Losses for {model.name}", save_name=f"{simulation_name}_all_losses", xlabel='Step', ylabel='Loss', mark_epochs=True, epoch_steps=epoch_steps)