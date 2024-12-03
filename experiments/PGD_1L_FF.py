import experiment_base
from experiment_base import run_experiment, GPT_CONFIG, GDM_CONFIG, PGD_CONFIG

config = PGD_CONFIG
config.use_ff = True
run_experiment(config)