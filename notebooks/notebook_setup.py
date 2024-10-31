import sys
import os
import subprocess

# Connect to colab if needed
try:
  from google.colab import drive
  drive.mount('/content/drive')
  os.chdir('/content/drive/My Drive/Research/llm-research')
  print("Connected to google drive")
except:
  print("Not connected to google drive")
  
# Set the sys path
path = os.path.abspath(os.path.join(os.getcwd(), '../'))
if path not in sys.path:
  sys.path.append(path)