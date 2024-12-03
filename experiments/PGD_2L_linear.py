import experiment_base
from experiment_base import run_experiment, GPT_CONFIG, GDM_CONFIG, PGD_CONFIG

config = PGD_CONFIG
config.n_layer = 2
config.attn_kernel_fn = 'linear'
run_experiment(config)