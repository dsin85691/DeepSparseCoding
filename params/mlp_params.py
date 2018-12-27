import os
from params.base_params import Base_Params

class params(Base_Params):
  def __init__(self):
    """
    Additional modifiable parameters:
      rectify_a    [bool] If set, rectify layer 1 activity
      norm_a       [bool] If set, l2 normalize layer 1 activity
      norm_weights [bool] If set, l2 normalize weights after updates
      batch_size   [int] Number of images in a training batch
      num_pixels   [int] Number of pixels
      num_hidden   [int] Number of layer 1 elements (# hidden units)
      num_classes  [int] Number of layer 2 elements (# categories)
      num_val      [int] Number of validation images
      val_on_cp    [bool] If set, compute validation performance on checkpoint
    """
    super(params, self).__init__()
    self.model_type = "MLP"
    self.model_name = "mlp"
    self.version = "0.0"
    self.optimizer = "annealed_sgd"
    self.vectorize_data = True
    self.rectify_a = True
    self.norm_a = False
    self.norm_weights = True
    self.num_batches = 4000
    self.batch_size = 100
    self.num_classes = 10
    self.num_hidden = 400
    self.num_val = 10000
    self.num_labeled = 50000
    self.cp_int = 100
    self.max_cp_to_keep = 1
    self.val_on_cp = True
    self.cp_load = False
    self.cp_load_name = "pretrain"
    self.cp_load_step = None
    self.cp_load_ver = "0.0"
    self.cp_load_var = ["w1"]
    self.log_int = 10
    self.log_to_file = True
    self.gen_plot_int = 100
    self.save_plots = True
    self.schedule = [
      {"weights": ["w1", "w2", "bias1", "bias2"],
      "weight_lr": [0.01, 0.001, 0.01, 0.001],
      "decay_steps": [int(self.num_batches*0.5)]*4,
      "decay_rate": [0.8]*4,
      "staircase": [True]*4}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type):
    self.model_name = "test_"+self.model_name
    self.set_data_params(data_type)
