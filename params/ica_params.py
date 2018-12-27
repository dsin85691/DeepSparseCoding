import os
from params.base_params import Base_Params

class params(Base_Params):
  def __init__(self):
    """
    Additional modifiable parameters:
      prior      [str] Prior for ICA - can be "laplacian" or "cauchy"
      batch_size [int] Number of images in a training batch
      num_pixels [int] Number of pixels
    """
    super(params, self).__init__()
    self.model_type = "ica"
    self.model_name = "ica"
    self.version = "0.0"
    self.num_images = 100
    self.vectorize_data = True
    self.norm_data = False
    self.center_data = False
    self.standardize_data = False
    self.contrast_normalize = False
    self.whiten_data = True
    self.whiten_method = "ZCA"
    self.lpf_data = True # only for ZCA/PCA
    self.lpf_cutoff = 0.7
    self.extract_patches = True
    self.num_patches = 1e6
    self.patch_edge_size = 16
    self.overlapping_patches = True
    self.randomize_patches = True
    self.patch_variance_threshold = 0
    self.num_batches = int(5e5)
    self.batch_size = 50
    self.prior = "laplacian" #"cauchy"
    self.optimizer = "annealed_sgd"
    self.cp_int = 1000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.cp_load_name = "pretrain"
    self.cp_load_step = None
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["w_synth"]
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 1000
    self.save_plots = True
    self.schedule = [
      {"weights": ["w_synth"], # A, VS265
      #{"weights": ["w_analysis"], # W=A^-1, Bell & Sejnowski
      "weight_lr": [0.001],
      "decay_steps": [5e5*0.8],
      "decay_rate": [0.8],
      "staircase": [True]}]
