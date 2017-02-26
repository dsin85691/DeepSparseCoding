import os
params = {
  "model_type": "lca",
  "model_name": "test",
  "out_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/MNIST/",
  "version": "0.0",
  "optimizer": "annealed_sgd",
  "rectify_a": False,
  "norm_images": False,
  "norm_weights": True,
  "batch_size": 100,
  "num_pixels": 784,
  "num_neurons": 400,
  "num_steps": 20,
  "dt": 0.001,
  "tau": 0.03,
  "thresh_type": "hard",
  "cp_int": 5000,
  "max_cp_to_keep": 5,
  "cp_load": False,
  "cp_load_name": "pretrain",
  "cp_load_val": 150000,
  "cp_load_ver": "0.0",
  "cp_load_var": ["phi"],
  "log_int": 100,
  "log_to_file": True,
  "gen_plot_int": 100,
  "display_plots": False,
  "save_plots": True,
  "eps": 1e-12,
  "device": "/cpu:0",
  "rand_seed": 1234567890}

schedule = [
  {"weights": ["phi"],
  "sparse_mult": 0.1,
  "weight_lr": [0.01],
  "decay_steps": [2000],
  "decay_rate": [0.8],
  "staircase": [True],
  "num_batches": 2000}]
