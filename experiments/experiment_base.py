import os
import sys
import subprocess

# Add parent directory to path

path = os.path.abspath(os.path.join(os.getcwd(), '../'))
if path not in sys.path:
  sys.path.append(path)
  
# Install requirements
    
try:
  subprocess.check_call(["pip", "install", "-r", "../requirements.txt", "--quiet"])
  print("Requirements installed")
except:
  print("Failed to install requirements")
  
# Paths
  
FIGURES_PATH = os.path.join(path, 'experiments/figures')
MODELS_PATH = os.path.join(path, 'experiments/models')
LOSSES_PATH = os.path.join(path, 'experiments/losses')
  
# Basic model initializations

