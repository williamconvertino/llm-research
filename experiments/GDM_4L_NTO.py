import experiment_base
from experiment_base import train_GPT, train_GDM, GPT_CONFIG, GDM_CONFIG

config = GDM_CONFIG
config.n_layer = 4
config.next_target_only = True
train_GDM(config)