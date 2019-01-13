import os
from params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    """
    Additional modifiable parameters:
      rectify_a    [bool] If set, rectify layer 1 activity
      norm_weights [bool] If set, l2 normalize weights after updates
      batch_size   [int] Number of images in a training batch
      num_neurons  [int] Number of LCA neurons
      num_steps    [int] Number of inference steps
      dt           [float] Discrete global time constant
      tau          [float] LCA time constant
      thresh_type  [str] "hard" or "soft" - LCA threshold function specification
    """
    super(params, self).__init__()
    self.model_type = "mlp_lca"
    self.model_name = "mlp_lca"
    self.version = "0.0"
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
    self.batch_size = 100
    # LCA Params
    self.num_neurons = 768
    self.num_steps = 50
    self.dt = 0.001
    self.tau = 0.03
    self.rectify_a = True
    self.norm_weights = True
    self.thresh_type = "soft"
    self.optimizer = "annealed_sgd"
    # MLP Params
    self.train_on_recon = True # if False, train on LCA latent activations
    self.num_classes = 10
    self.layer_types = ["fc", "fc", "fc"]
    self.output_channels = [300, 500, self.num_classes]
    self.patch_size_y = [None, None, None]
    self.patch_size_x = [None, None, None]
    self.conv_strides = [None, None, None]
    self.batch_norm = [None, None, None]
    self.dropout = [1.0, 1.0, 1.0]
    self.max_pool = [False, False, False]
    self.max_pool_ksize = [None, None, None]
    self.max_pool_strides = [None, None, None]
    # Others
    self.cp_int = 10000
    self.val_on_cp = True
    self.max_cp_to_keep = None
    self.cp_load = True
    self.cp_load_name = "lca_mnist"
    self.cp_load_step = None # latest checkpoint
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["weights/w:0"]
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 10000
    self.save_plots = True
    self.schedule = [
      #Training LCA
      #{"weights": None, #["weights/w:0"],
      #"train_lca": True,
      #"num_batches": int(1e4),
      #"sparse_mult": 0.1,
      #"weight_lr": 0.01,#[0.01],
      #"decay_steps": int(1e4*0.5),#[int(1e4*0.5)],
      #"decay_rate": 0.8,#[0.8],
      #"staircase": True},
      #Training MLP on LCA recons
      #Only training MLP weights, not VAE
      #TODO change weight names
      {"weights": [
        "layer0/conv_w_0:0",
        "layer0/conv_b_0:0",
        "layer1/conv_w_1:0",
        "layer1/conv_b_1:0",
        "layer2/fc_w_2:0",
        "layer2/fc_b_2:0",
        "layer3/fc_w_3:0",
        "layer3/fc_b_3:0"
        ],
      "train_lca": False,
      "num_batches": int(1e4),
      "sparse_mult": 0.01,
      "weight_lr": 0.01,
      "decay_steps": int(1e4*0.8),
      "decay_rate": 0.8,
      "staircase": True},
      ]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.vectorize_data = True
      self.rescale_data = True
      self.center_data = False
      self.whiten_data = False
      self.extract_patches = False
      # LCA params
      self.num_neurons = 768
      # MLP params
      self.train_on_recon = True # if False, train on activations
      self.full_data_shape = [28, 28, 1]
      self.num_classes = 10
      self.optimizer = "adam"
      self.layer_types = ["conv", "conv", "fc", "fc"]
      self.output_channels = [32, 64, 1024, self.num_classes]
      self.patch_size_y = [5, 5, None, None]
      self.patch_size_x = self.patch_size_y
      self.conv_strides = [(1,1,1,1), (1,1,1,1), None, None]
      self.batch_norm = [None, None, None, None]
      self.dropout = [1.0, 1.0, 0.4, 1.0]
      self.max_pool = [True, True, False, False]
      self.max_pool_ksize = [(1,2,2,1), (1,2,2,1), None, None]
      self.max_pool_strides = [(1,2,2,1), (1,2,2,1), None, None]
      # NOTE schedule index will change if lca training is happening
      self.schedule[0]["num_batches"] = int(2e4)
      self.schedule[0]["sparse_mult"] = 0.21
      self.schedule[0]["weight_lr"] = 1e-4
      self.schedule[0]["decay_steps"] = int(0.8*self.schedule[0]["num_batches"])
      self.schedule[0]["decay_rate"] = 0.90

    elif data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 16
      self.vectorize_data = True
      self.rescale_data = True
      self.whiten_data = False
      self.extract_patches = False
      self.num_neurons = 768
      self.train_on_recon = True # if False, train on activations
      self.num_classes = 2
      self.output_channels = [128, 64, self.num_classes]
      for sched_idx in range(len(self.schedule)):
        self.schedule[sched_idx]["sparse_mult"] = 0.21
        self.schedule[sched_idx]["weight_lr"] = 0.1
        self.schedule[sched_idx]["num_batches"] = int(1e5)
        self.schedule[sched_idx]["decay_steps"] = int(0.8*self.schedule[sched_idx]["num_batches"])

    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type):
    self.set_data_params(data_type)
    self.epoch_size = 50
    self.batch_size = 10
    self.num_edge_pixels = 8
    for sched_idx in range(len(self.schedule)):
      self.schedule[sched_idx]["weights"] = None
      self.schedule[sched_idx]["num_batches"] = 2
      self.schedule[sched_idx]["weight_lr"] = 1e-4
    self.num_neurons = 100
    self.num_steps = 5