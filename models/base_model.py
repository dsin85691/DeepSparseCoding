import os
import logging
import numpy as np
import tensorflow as tf

class Model(object):
  def __init__(self, params, schedule):
    self.optimizers_added = False
    self.savers_constructed = False
    self.load_schedule(schedule)
    self.sched_idx = 0
    self.load_params(params)
    self.make_dirs()
    self.init_logging()
    self.log_params(params)

  def setup_graph(self, graph):
    self.graph = graph
    self.add_optimizers_to_graph()
    self.add_initializer_to_graph()
    self.construct_savers()

  """
  Load schedule into object
  Inputs:
   schedule: [list of dict] learning schedule
  """
  def load_schedule(self, schedule):
    for sched in schedule:
      assert len(sched["weights"]) == len(sched["weight_lr"])
      assert len(sched["weights"]) == len(sched["decay_steps"])
      assert len(sched["weights"]) == len(sched["decay_rate"])
      assert len(sched["weights"]) == len(sched["staircase"])
    self.sched = schedule

  """
  Load parameters into object
  Inputs:
   params: [dict] model parameters
  Modifiable Parameters:
    model_type     [str] Type of model
                         Can be "MLP", "karklin_lewicki"
    model_name     [str] Name for model
    output_dir     [str] Base directory where output will be directed
    version        [str] Model version for output
    optimizer      [str] Which optimization algorithm to use
                         Can be "annealed_sgd" or "adam"
    log_int        [int] How often to send updates to stdout
    log_to_file    [bool] If set, log to file, else log to stderr
    gen_plot_int   [int] How often to generate plots
    display_plots  [bool] If set, display plots
    save_plots     [bool] If set, save plots to file
    cp_int         [int] How often to checkpoint
    val_on_cp      [bool] If set, compute validation performance on checkpoint
    cp_load        [bool] if set, load from checkpoint
    cp_load_name   [str] checkpoint model name to load
    cp_load_val    [int] checkpoint time step to load
    cp_load_ver    [str] checkpoint version to load
    cp_load_var    [list of str] which variables to load
                         if None or empty list, the model load all weights
    eps            [float] Small value to avoid division by zero
    device         [str] Which device to run on
    rand_seed      [int] Random seed
  """
  def load_params(self, params):
    # Meta-parameters
    self.model_name = str(params["model_name"])
    self.model_type = str(params["model_type"])
    if "num_labeled" in params.keys():
      self.num_labeled = str(params["num_labeled"])
    if "num_unlabeled" in params.keys():
      self.num_unlabeled = str(params["num_unlabeled"])
    self.version = str(params["version"])
    self.optimizer = str(params["optimizer"])
    # Output generation
    self.log_int = int(params["log_int"])
    self.log_to_file = bool(params["log_to_file"])
    self.gen_plot_int = int(params["gen_plot_int"])
    self.disp_plots = bool(params["display_plots"])
    self.save_plots = bool(params["save_plots"])
    # Checkpointing
    if "val_on_cp" in params.keys():
      self.val_on_cp = bool(params["val_on_cp"])
    else:
      self.val_on_cp = False
    self.cp_int = int(params["cp_int"])
    self.cp_load = bool(params["cp_load"])
    self.cp_load_name = str(params["cp_load_name"])
    self.cp_load_val = int(params["cp_load_val"])
    self.cp_load_ver = str(params["cp_load_ver"])
    if params["cp_load_var"]:
      self.cp_load_var = [str(var) for var in params["cp_load_var"]]
    else:
      self.cp_load_var = []
    # Directories
    self.out_dir = str(params["output_dir"]) + self.model_name
    self.cp_save_dir = self.out_dir + "/checkpoints/"
    self.cp_load_dir = (str(params["output_dir"]) + self.cp_load_name
      + "/checkpoints/")
    self.log_dir = self.out_dir + "/logfiles/"
    self.save_dir = self.out_dir + "/savefiles/"
    self.disp_dir = self.out_dir + "/vis/"
    self.analysis_dir = self.out_dir + "/analysis/"
    # Other
    self.eps = float(params["eps"])
    self.device = str(params["device"])
    self.rand_seed = int(params["rand_seed"])

  """
  Get param value from model
    equivalent to self.param_name
  """
  def get_param(self, param_name):
    assert hasattr(self, param_name)
    return getattr(self, param_name)

  """
  Modifies a model parameter
  Inputs:
    param_name: [str] parameter name, must already exist
    new_value: [] new parameter value, must be the same type as old param value
  """
  def set_param(self, param_name, new_value):
    assert hasattr(self, param_name)
    assert type(getattr(self, param_name)) == type(new_value)
    setattr(self, param_name, new_value)

  """Make output directories"""
  def make_dirs(self):
    if not os.path.exists(self.out_dir):
      os.makedirs(self.out_dir)
    if not os.path.exists(self.log_dir):
      os.makedirs(self.log_dir)
    if not os.path.exists(self.cp_save_dir):
      os.makedirs(self.cp_save_dir)
    if not os.path.exists(self.save_dir):
      os.makedirs(self.save_dir)
    if not os.path.exists(self.disp_dir):
      os.makedirs(self.disp_dir)
    if not os.path.exists(self.analysis_dir):
      os.makedirs(self.analysis_dir)

  """Logging to track run statistics"""
  def init_logging(self):
    logging_level = logging.INFO
    log_format = ("%(asctime)s.%(msecs)03d"
      +" -- %(message)s")
    date_format = "%m-%d-%Y %H:%M:%S"
    if self.log_to_file:
      log_filename = self.log_dir+self.model_name+"_v"+self.version+".log"
      logging.basicConfig(filename=log_filename, format=log_format,
        datefmt=date_format, filemode="w", level=logging.INFO)
    else:
      logging.basicConfig(format=log_format, datefmt=date_format, filemode="w",
        level=logging.INFO)

  """Use logging to write model params"""
  def log_params(self, params):
    for key in sorted(params.keys()):
     logging.info("param: "+key+" = "+str(params[key])+" "
       +str(type(params[key])))

  """Use logging to write current schedule, as specified by self.sched_idx"""
  def log_current_schedule(self):
    current_schedule = self.sched[self.sched_idx]
    for key in sorted(current_schedule.keys()):
     logging.info("sched_"+str(self.sched_idx)+": "+key+" = "
       +str(current_schedule[key])+" "+str(type(current_schedule[key])))

  """Use logging to print input string to stderr"""
  def log_info(self, string):
    logging.info(str(string))

  """
  Add optimizers to graph
  Creates member variables grads_and_vars and apply_grads for each weight
    - both member variables are indexed by [schedule_idx][weight_idx]
    - grads_and_vars holds the gradients for the weight updates
    - apply_grads is the operator to be called to perform weight updates
  """
  def add_optimizers_to_graph(self):
    with self.graph.as_default():
      with tf.name_scope("optimizers") as scope:
        self.grads_and_vars = list()
        self.apply_grads = list()
        for schedule_idx, sch in enumerate(self.sched):
          sch_grads_and_vars = list()
          sch_apply_grads = list()
          for w_idx, weight in enumerate(sch["weights"]):
            learning_rates = tf.train.exponential_decay(
              learning_rate=sch["weight_lr"][w_idx],
              global_step=self.global_step,
              decay_steps=sch["decay_steps"][w_idx],
              decay_rate=sch["decay_rate"][w_idx],
              staircase=sch["staircase"][w_idx],
              name="annealing_schedule_"+weight)
            if self.optimizer == "annealed_sgd":
              optimizer = tf.train.GradientDescentOptimizer(learning_rates,
                name="grad_optimizer_"+weight)
            elif self.optimizer == "adam":
              optimizer = tf.train.AdamOptimizer(learning_rates,
                beta1=0.9, beta2=0.99, epsilon=1e-07,
                name="adam_optimizer_"+weight)
            with tf.variable_scope("weights", reuse=True) as scope:
              weight_var = [tf.get_variable(weight)]
            sch_grads_and_vars.append(
              optimizer.compute_gradients(self.total_loss, var_list=weight_var))
            if w_idx == 0: # Only want to update global step once
              sch_apply_grads.append(
                optimizer.apply_gradients(sch_grads_and_vars[w_idx],
                global_step=self.global_step))
            else:
              sch_apply_grads.append(
                optimizer.apply_gradients(sch_grads_and_vars[w_idx],
                global_step=None))
          self.grads_and_vars.append(sch_grads_and_vars)
          self.apply_grads.append(sch_apply_grads)
    self.optimizers_added = True

  """Add initializer to the graph
  This must be done after optimizers have been added
  """
  def add_initializer_to_graph(self):
    assert self.optimizers_added
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("initialization") as scope:
          self.init_op = tf.initialize_all_variables()

  """Get variables for loading"""
  def get_load_vars(self):
    v = tf.all_variables()
    if len(self.cp_load_var) > 0:
      v = [var
        for var in v
        for weight in self.cp_load_var
        if weight in var.name]
    return v

  """Add savers to graph"""
  def construct_savers(self):
    assert self.optimizers_added, (
      "Optimizers must be added to the graph before constructing savers.")
    with self.graph.as_default():
      with tf.variable_scope("weights", reuse=True) as scope:
        weights = [weight for weight in tf.all_variables()
          if weight.name.startswith(scope.name)]
      self.weight_saver = tf.train.Saver(var_list=weights)
      self.full_saver = tf.train.Saver()
      if self.cp_load and len(self.cp_load_var) > 0:
        self.loader = tf.train.Saver(var_list=self.get_load_vars())
    self.savers_constructed = True

  """Write saver definitions for full model and weights-only"""
  def write_saver_defs(self):
    assert self.savers_constructed
    full_saver_def = self.full_saver.as_saver_def()
    full_file = self.save_dir+self.model_name+"_v"+self.version+"-full.def"
    with open(full_file, "wb") as f:
        f.write(full_saver_def.SerializeToString())
    logging.info("Full saver def saved in file %s"%full_file)
    weight_saver_def = self.weight_saver.as_saver_def()
    weight_file = self.save_dir+self.model_name+"_v"+self.version+"-weights.def"
    with open(weight_file, "wb") as f:
        f.write(weight_saver_def.SerializeToString())
    logging.info("Weight saver def saved in file %s"%weight_file)

  """Write graph structure to protobuf file"""
  def write_graph(self, graph_def):
    write_name = self.model_name+"_v"+self.version+".pb"
    self.writer = tf.train.SummaryWriter(self.save_dir, graph=self.graph)
    tf.train.write_graph(graph_def,
      logdir=self.save_dir, name=write_name, as_text=False)
    logging.info("Graph def saved in file %s"%self.save_dir+write_name)

  """Write checkpoints for full model and weights-only"""
  def write_checkpoint(self, session):
    base_save_path = self.cp_save_dir+self.model_name+"_v"+self.version
    full_save_path = self.full_saver.save(session,
      save_path=base_save_path+"_full", global_step=self.global_step)
    logging.info("Full model saved in file %s"%full_save_path)
    weight_save_path = self.weight_saver.save(session,
      save_path=base_save_path+"_weights",
      global_step=self.global_step)
    logging.info("Weights model saved in file %s"%weight_save_path)
    return base_save_path

  """
  Returns the current schedule being executed
  Inputs:
    key: [str] key in dictionary
  """
  def get_sched(self, key=None):
    if key:
      assert key in self.sched[self.sched_idx].keys(), (
        key+" must be in the schedule.")
      return self.sched[self.sched_idx][key]
    return self.sched[self.sched_idx]

  """
  Modifies the internal schedule for the current schedule index
  Inputs:
    key: [str] key in dictionary
    val: value be set in schedlue,
      if there is already a val for key, new val must be of same type
  """
  def set_sched(self, key, val):
    if key in self.sched[self.sched_idx].keys():
      assert type(val) == type(self.sched[self.sched_idx][key]), (
        "val must have type "+str(type(self.sched[self.sched_idx][key])))
    self.sched[self.sched_idx][key] = val

  """
  Load checkpoint weights into session.
  Inputs:
    session: tf.Session() that you want to load into
    model_dir: String specifying the path to the checkpoint
  """
  def load_model(self, session, model_dir):
    self.full_saver.restore(session, model_dir)

  """
  Return dictionary containing all placeholders
  Inputs:
    input_data: data to be placed in self.s
    input_label: label to be placed in self.y
  """
  def get_feed_dict(self, input_data, input_label=None):
    skip_num = 2 if input_label is not None else 1
    placeholders = [op.name
      for op
      in self.graph.get_operations()
      if "placeholders" in op.name][skip_num:]
    if input_label is not None:
      feed_dict = {self.x:input_data, self.y:input_label}
    else:
      feed_dict = {self.x:input_data}
    for placeholder in placeholders:
      feed_dict[self.graph.get_tensor_by_name(placeholder+":0")] = (
        self.get_sched(placeholder.split('/')[1]))
    return feed_dict

  """
  Log train progress information
  Inputs:
    input_data: data object containing the current image batch
    input_label: data object containing the current label batch
    batch_step: current batch number within the schedule
  """
  def print_update(self, input_data, input_label=None, batch_step=0):
    pass

  """
  Plot weights, reconstruction, gradients, etc
  Inputs:
    input_data: data object containing the current image batch
    input_label: data object containing the current label batch
  """
  def generate_plots(self, input_data, input_label=None):
    pass
