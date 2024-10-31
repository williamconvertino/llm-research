import os

try:
  from google.colab import drive
  drive.mount('/content/drive')
  os.chdir('/content/drive/My Drive/Research/llm-research')
  print("Connected to google drive")
except:
  print("Not connected to google drive")