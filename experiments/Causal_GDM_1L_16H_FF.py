import experiment_base
from experiment_base import run_experiment, GPT_CONFIG, GDM_CONFIG, CAUSAL_GDM_CONFIG

config = CAUSAL_GDM_CONFIG
config.use_ff = True
config.n_head = 16
run_experiment(config)