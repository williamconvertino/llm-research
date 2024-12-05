import experiment_base
from experiment_base import run_experiment, GPT_CONFIG, GDM_CONFIG, CAUSAL_GDM_CONFIG, CAUSAL_GPT_CONFIG

config = CAUSAL_GPT_CONFIG
config.use_ff = True
run_experiment(config)