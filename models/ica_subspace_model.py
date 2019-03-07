import numpy as np 
import tensorflow as tf 
#import utils.plot_functions as pf 
#import utils.data_processing as dp 
from models.ica_model import IcaModel

class IcaSubspaceModel(IcaModel):

  def __init__(self):
    super(IcaSubspaceModel, self).__init__()

  def load_params(self, params):
    """Parameters are the same as classical ICA, with addition of number of subspace/groups and
    group sizes for each subspace.
    
    Variables:
      num_groups: (int) number of groups/subspaces
      group_sizes: (int[]) number of vectors at each subspace. ith index for ith group.
      group_index: (int[]) index of vectors for each subspace
    

    """
    super(IcaSubspaceModel, self).load_params(params)
    self.input_shape = [None, self.num_neurons]

    # new params for subspace ica
    self.num_groups = self.params.num_groups
    self.group_sizes = self.construct_group_sizes(self.params.group_sizes)
    self.group_index = [sum(self.group_sizes[:i]) for i in range(self.num_groups)]
    print("group_index", self.group_index)
    self.sum_arr = self.construct_sum_arr() 
    print("sumarr shape", self.sum_arr.shape)


  def get_input_shape(self):
      return self.input_shape


  def build_graph_from_input(self, input_node):
    """Build the Tensorflow graph object. 

    Placeholders:
      input_img: (float[]) input image patch

    Variables:
      w_synth: (float[][]) synthesis weights; basis; "A" in classical ICA
      w_analy: (float[][]) analysis weights; inverse basis; "A.T" in classical ICA (since A is orthonormal)
      s: (float[]) latent variables, computed from inner product of w_analy and input_img
      recon: (float[]) recontruction of image patch using w_synth and s

    """
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.input_img = tf.placeholder(tf.float32, shape=self.input_shape, name="input_data")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("weights") as scope:
          Q, R = np.linalg.qr(np.random.standard_normal(self.w_analysis_shape))
          self.w_synth = tf.get_variable(name="w_synth", dtype=tf.float32, initializer=Q.astype(np.float32), trainable=True)
          self.w_analy = tf.transpose(self.w_synth, name="w_analy")
          self.trainable_variables[self.w_synth.name] = self.w_synth

        with tf.name_scope("inference") as scope:
          self.s = tf.matmul(tf.transpose(self.w_analy), self.input_img, name="latent_variables") 

        with tf.name_scope("output") as scope:
          with tf.name_scope("image_estimate"):
            self.recon = tf.matmul(self.w_synth, self.s, name="reconstruction")

    self.graph_built = True

  def compute_weight_gradients(self, optimizer, weight_op=None):
    def nonlinearity(u):
      return u**(-0.5)

    if(type(weight_op) is not list):
      weight_op = [weight_op]

    assert len(weight_op) == 1, ("IcaModel should only have one weight matrix")

    print("weight shape", weight_op[0].shape)
    print("input_img.shape", self.input_img.shape)
    
#    wI = tf.matmul(tf.transpose(weight_op[0]), self.input_img)
    wI = tf.transpose(tf.matmul(self.input_img, weight_op[0]))
    print("wI.shape", wI.shape)

    group_scalars = tf.matmul(tf.transpose(tf.math.pow(wI, 2)), self.sum_arr) # [1, num_groups]
    print("group_scalars.shape", group_scalars.shape)
    
    nonlinear_term = nonlinearity(tf.matmul(self.sum_arr, tf.transpose(group_scalars))) 
    print("nonlienar_term.shape", nonlinear_term.shape)

    scalars = tf.math.multiply(wI, nonlinear_term)
    print("scalars.shape", scalars.shape)

    img_tiled = tf.tile(self.input_img, [self.num_neurons, 1])
    print("img_tiled.shape", img_tiled.shape)
    
    gradient = tf.transpose(tf.matmul(tf.transpose(img_tiled), scalars), name="gradient")
    print("gradient.shape", gradient.shape)

    #gradient = tf.zeros_like(weight_op[0])
    #print("CORRECT gradient shape", gradient.shape)

    return [(gradient, weight_op[0])]

  def compute_weight_gradients1(self, optimizer, weight_op=None):
      def nonlinearity(u):
        return tf.power(u, -0.5)
    
      grads = []

      for i in range(self.input_shape[0]):
        wI = tf.matmul(tf.transpose(weight_op[0], self.input_img[i]))
        group_scalars = tf.matmul(tf.transpose(tf.math.pow(wI, 2)), self.sum_arr)
        nonlinear_term = nonlinearity(tf.matmul(self.sum_arr, tf.tranpose(group_scalars)))
        scalars = tf.math.multiply(wI, nonlinear_term)
        img_tiled = tf.tile(self.input_img[i], [1, self.num_nuerons])
        gradient = tf.tranpose(tf.matmul(tf.tranpose(img_tiled), scalars))
        grads.append(gradient)
      avg_grads = tf.reduce_mean(grads, axis=0, name="gradient")
      return [(avg_grads, weight_op[0])]


    
  def construct_group_sizes(self, params_group_sizes):
    """Construct respective group sizes. If group_size initialzed as None, then group sizes are uniformally
    distributed; unless specified otherwise. """
    self.group_sizes = params_group_sizes
    if params_group_sizes is None:
      self.group_sizes = [self.num_neurons // self.num_groups for _ in range(self.num_groups)]
    
    assert sum(self.group_sizes) == self.num_neurons, ("Total number of vectors should be the same "
                                                        "as number of neurons.")
    print("construct_group_sizes: {}".format(self.group_sizes))
    return self.group_sizes

  def construct_sum_arr(self):
    sum_arr = []
    for s, i in zip(self.group_sizes, self.group_index):
      col_index = np.zeros(self.num_neurons)
      col_index[i:i+s] = 1
      sum_arr.append(col_index)
    sum_arr = np.stack(sum_arr, axis=1)
    sum_arr = np.float32(sum_arr)
    return sum_arr

  def get_subspace(g):
    """Return the column vectors in the g-th subspace. """
    num_vec = self.group_sizes[g]
    subspace_index = self.group_index[g]
    return self.w_synth[:, subspace_index:subspace_index+num_vec]
    

    

    
