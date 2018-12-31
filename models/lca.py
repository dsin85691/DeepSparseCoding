import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
from models.base_model import Model

class LCA(Model):
  def __init__(self):
    super(LCA, self).__init__()

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(LCA, self).load_params(params)
    # Network Size
    self.batch_size = int(params.batch_size)
    self.num_pixels = int(np.prod(self.params.data_shape))
    self.num_neurons = int(params.num_neurons)
    self.phi_shape = [self.num_pixels, self.num_neurons]
    self.u_shape = [self.num_neurons]
    self.x_shape = [None, self.num_pixels]
    # Hyper Parameters
    self.eta = self.params.dt / self.params.tau

  def compute_excitatory_current(self):
    return tf.matmul(self.x, self.phi, name="driving_input")

  def compute_inhibitory_connectivity(self):
   return (tf.matmul(tf.transpose(self.phi), self.phi, name="gram_matrix")
     - tf.constant(np.identity(self.phi_shape[1], dtype=np.float32), name="identity_matrix"))

  def threshold_units(self, u_in):
    if self.params.thresh_type == "soft":
      if self.params.rectify_a:
        a_out = tf.where(tf.greater(u_in, self.sparse_mult),
          tf.subtract(u_in, self.sparse_mult), self.u_zeros)
      else:
        a_out = tf.where(tf.greater_equal(u_in, self.sparse_mult),
          tf.subtract(u_in, self.sparse_mult),
          tf.where(tf.less_equal(u_in, -self.sparse_mult),
          tf.add(u_in, self.sparse_mult),
          self.u_zeros))
    elif self.params.thresh_type == "hard":
      if self.params.rectify_a:
        a_out = tf.where(tf.greater(u_in, self.sparse_mult), u_in, self.u_zeros)
      else:
        a_out = tf.where(tf.greater(u_in, self.sparse_mult), u_in,
          tf.where(tf.less(u_in, -self.sparse_mult), u_in, self.u_zeros))
    else:
      a_out = tf.identity(u_in)
    return a_out

  def step_inference(self, u_in, a_in, b, g, step):
    with tf.name_scope("update_u"+str(step)) as scope:
      lca_explain_away = tf.matmul(a_in, g, name="explaining_away")
      du = tf.subtract(tf.subtract(b, lca_explain_away), u_in, name="du")
      u_out = tf.add(u_in, tf.multiply(self.eta, du))
    return u_out, lca_explain_away

  def infer_coefficients(self):
   lca_b = self.compute_excitatory_current()
   lca_g = self.compute_inhibitory_connectivity()
   u_list = [self.u_zeros]
   a_list = [self.threshold_units(u_list[0])]
   for step in range(self.params.num_steps-1):
     u = self.step_inference(u_list[step], a_list[step], lca_b, lca_g, step)[0]
     u_list.append(u)
     a_list.append(self.threshold_units(u_list[step+1]))
   return (u_list, a_list)

  def compute_recon(self, a_in):
    return tf.matmul(a_in, tf.transpose(self.phi), name="reconstruction")

  def compute_recon_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      reduc_dim = list(range(1, len(a_in.shape))) # Want to avg over batch, sum over the rest
      recon_loss = tf.reduce_mean(0.5 *
        tf.reduce_sum(tf.square(tf.subtract(self.x, self.compute_recon(a_in))),
        axis=reduc_dim), name="recon_loss")
    return recon_loss

  def compute_sparse_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      reduc_dim = list(range(1, len(a_in.shape))) # Want to avg over batch, sum over the rest
      sparse_loss = self.sparse_mult * tf.reduce_mean(tf.reduce_sum(tf.abs(a_in),
        axis=reduc_dim), name="sparse_loss")
    return sparse_loss

  def compute_total_loss(self, a_in, loss_funcs):
    """
    Returns sum of all loss functions defined in loss_funcs for given a_in
    Inputs:
      a_in [tf.Variable] containing the sparse coding activity values
      loss_funcs [dict] containing keys that correspond to names of loss functions and values that
        point to the functions themselves
    """
    total_loss = tf.add_n([func(a_in) for func in loss_funcs.values()], name="total_loss")
    return total_loss

  def get_loss_funcs(self):
    return {"recon_loss":self.compute_recon_loss, "sparse_loss":self.compute_sparse_loss}

  def build_graph(self):
    super(LCA, self).build_graph()
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")
          self.sparse_mult = tf.placeholder(tf.float32, shape=(), name="sparse_mult")

        with tf.name_scope("constants") as scope:
          u_full_shape = tf.stack([tf.shape(self.x)[0]]+self.u_shape)
          self.u_zeros = tf.zeros(shape=u_full_shape, dtype=tf.float32, name="u_zeros")
          self.u_noise = tf.truncated_normal(shape=u_full_shape, mean=0.0, stddev=0.1,
            dtype=tf.float32, name="u_noise")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        phi_norm_dim = list(range(len(self.phi_shape)-1)) # normalize across input dim(s)

        with tf.variable_scope("weights") as scope:
          self.weight_scope = tf.get_variable_scope()
          phi_init = tf.nn.l2_normalize(tf.truncated_normal(self.phi_shape, mean=0.0,
            stddev=0.5, dtype=tf.float32), axis=phi_norm_dim, epsilon=self.params.eps, name="phi_init")
          self.phi = tf.get_variable(name="phi", dtype=tf.float32, initializer=phi_init,
            trainable=True)
          self.trainable_variables[self.phi.name] = self.phi

        with tf.name_scope("norm_weights") as scope:
          self.norm_phi = self.phi.assign(tf.nn.l2_normalize(self.phi, axis=phi_norm_dim,
            epsilon=self.params.eps, name="row_l2_norm"))
          self.norm_weights = tf.group(self.norm_phi, name="l2_normalization")

        with tf.variable_scope("inference") as scope:
          self.inference_scope = tf.get_variable_scope()
          u_list, a_list = self.infer_coefficients()
          self.u = tf.identity(u_list[-1], name="u")
          self.a = tf.identity(a_list[-1], name="activity")

        with tf.name_scope("output") as scope:
          self.x_ = self.compute_recon(self.a)

        with tf.name_scope("loss") as scope:
          loss_funcs = self.get_loss_funcs()
          self.loss_dict = dict(zip(
            [key for key in loss_funcs.keys()], [func(self.a) for func in loss_funcs.values()]))
          self.total_loss = self.compute_total_loss(self.a, loss_funcs)

        with tf.name_scope("performance_metrics") as scope:
          with tf.name_scope("reconstruction_quality"):
            MSE = tf.reduce_mean(tf.square(tf.subtract(self.x, self.x_)), axis=[1, 0],
              name="mean_squared_error")
            pixel_var = tf.nn.moments(self.x, axes=[1])[1]
            # TODO: pSNRdB could possibly be infinity, need to check for that and set it to a cap
            self.pSNRdB = tf.multiply(10.0, tf.log(tf.divide(tf.square(pixel_var), MSE)),
              name="recon_quality")
    self.graph_built = True

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
    Log train progress information
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    update_dict = super(LCA, self).generate_update_dict(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.loss_dict["recon_loss"], self.loss_dict["sparse_loss"],
      self.total_loss, self.a, self.x_, self.pSNRdB]
    grad_name_list = []
    learning_rate_dict = {}
    for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
      eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name[1]]
      grad_name = weight_grad_var[0][1].name.split('/')[1].split(':')[0] #2nd is np.split
      grad_name_list.append(grad_name)
      learning_rate_dict[grad_name] = self.get_schedule("weight_lr")[w_idx]
    out_vals =  tf.get_default_session().run(eval_list, feed_dict)
    current_step, recon_loss, sparse_loss, total_loss, a_vals, recon, pSNRdB = out_vals[0:7]
    input_max = np.max(input_data)
    input_mean = np.mean(input_data)
    input_min = np.min(input_data)
    recon_max = np.max(recon)
    recon_mean = np.mean(recon)
    recon_min = np.min(recon)
    a_vals_max = np.array(a_vals.max())
    a_vals_mean = np.array(a_vals.mean())
    a_vals_min = np.array(a_vals.min())
    a_frac_act = np.array(np.count_nonzero(a_vals)
      / float(a_vals.size))
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "schedule_index":self.sched_idx,
      "recon_loss":recon_loss,
      "sparse_loss":sparse_loss,
      "total_loss":total_loss,
      "a_fraction_active":a_frac_act,
      "a_max_mean_min":[a_vals_max, a_vals_mean, a_vals_min],
      "x_max_mean_min":[input_max, input_mean, input_min],
      "x_hat_max_mean_min":[recon_max, recon_mean, recon_min]}
    grads = out_vals[7:]
    for grad, name in zip(grads, grad_name_list):
      grad_max = learning_rate_dict[name]*np.array(grad.max())
      grad_min = learning_rate_dict[name]*np.array(grad.min())
      grad_mean = learning_rate_dict[name]*np.mean(np.array(grad))
      stat_dict[name+"_grad_max_mean_min"] = [grad_max, grad_mean, grad_min]
    update_dict.update(stat_dict) #stat_dict overwrites for same keys
    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(LCA, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.phi, self.x_,  self.a]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    weights, recon, activity = eval_out[1:]
    weights_norm = np.linalg.norm(weights, axis=0, keepdims=False)
    recon = dp.reshape_data(recon, flatten=False)[0]
    weights = dp.reshape_data(weights.T, flatten=False)[0] # [num_neurons, height, width]
    fig = pf.plot_activity_hist(input_data, title="Image Histogram",
      save_filename=(self.params.disp_dir+"img_hist_"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
    input_data = dp.reshape_data(input_data, flatten=False)[0]
    fig = pf.plot_data_tiled(input_data, normalize=False,
      title="Images at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.params.disp_dir+"images_"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_activity_hist(activity, title="Activity Histogram",
      save_filename=(self.params.disp_dir+"act_hist_"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_data_tiled(weights, normalize=False,
      title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.params.disp_dir+"phi_v"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_data_tiled(recon, normalize=False,
      title="Recons at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.params.disp_dir+"recons_v"+self.params.version+"-"+current_step.zfill(5)+".png"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      grad = dp.reshape_data(grad.T, flatten=False)[0]
      fig = pf.plot_data_tiled(grad, normalize=True,
        title="Gradient for phi at step "+current_step, vmin=None, vmax=None,
        save_filename=(self.params.disp_dir+"dphi_v"+self.params.version+"_"+current_step.zfill(5)+".png"))
