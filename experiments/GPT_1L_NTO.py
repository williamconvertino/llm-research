import experiment_base
from experiment_base import run_experiment, GPT_CONFIG, GDM_CONFIG

config = GPT_CONFIG
config.use_nto = True
run_experiment(config)