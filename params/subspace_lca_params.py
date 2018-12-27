import os
from params.base_params import Base_Params

class params(Base_Params):
  def __init__(self):
    super(params, self).__init__()
    self.model_type = "subspace_lca"
    self.model_name = "subspace_lca"
    self.version = "0.0"
    self.num_images = 150
    self.vectorize_data = True
    self.norm_data = False
    self.center_data = True
    self.standardize_data = False
    self.contrast_normalize = False
    self.whiten_data = True
    self.whiten_method = "FT"
    self.lpf_data = False # only for ZCA
    self.lpf_cutoff = 0.7
    self.extract_patches = True
    self.num_patches = 1e6
    self.patch_edge_size = 16
    self.overlapping_patches = True
    self.randomize_patches = True
    self.patch_variance_threshold = 0.0
    self.num_batches = int(1e5)
    self.batch_size = 80
    self.num_neurons = 768
    self.num_groups = 192
    self.num_steps = 60
    self.dt = 0.001
    self.tau = 0.03
    self.norm_weights = True
    self.optimizer = "annealed_sgd"
    self.cp_int = 10000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.cp_load_name = "pretrain"
    self.cp_load_step = None # latest checkpoint
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["phi"]
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 5000
    self.save_plots = True
    self.schedule = [
      {"weights": ["phi"],
      "sparse_mult": 5.0,
      "group_orth_mult": 0.1,
      "weight_lr": [0.02],
      "decay_steps": [int(self.num_batches*0.8)],
      "decay_rate": [0.7],
      "staircase": [True]}]

  def set_data_params(data_type):
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.vectorize_data = True
      self.norm_data = False
      self.rescale_data = True
      self.center_data = False
      self.standardize_data = False
      self.contrast_normalize = False
      self.whiten_data = False
      self.lpf_data = False # only for ZCA
      self.lpf_cutoff = 0.7
      self.extract_patches = False
      self.num_neurons = 768
      self.num_groups = 128
      self.schedule["sparse_mult"] = 0.45
      self.schedule["group_orth_mult"] = 0.04
      self.schedule["weight_lr"] = 0.08

    elif data_type.lower() == "vanhateren":
      self.model_name += "_vh"
      self.num_images = 150
      self.vectorize_data = True
      self.norm_data = False
      self.rescale_data = False
      self.center_data = False
      self.standardize_data = False
      self.contrast_normalize = False
      self.whiten_data = True
      self.whiten_method = "FT"
      self.lpf_data = False # FT whitening already does LPF
      self.lpf_cutoff = 0.7
      self.extract_patches = True
      self.num_patches = 1e6
      self.patch_edge_size = 16
      self.overlapping_patches = True
      self.randomize_patches = True
      self.patch_variance_threshold = 0.0
      self.num_neurons = 768
      self.num_groups = 192
      self.schedule["sparse_mult"] = 5.0
      self.schedule["group_orth_mult"] = 0.1
      self.schedule["weight_lr"] = 0.02

    else:
      assert False, ("Data type "+data_type+" is not supported.")
