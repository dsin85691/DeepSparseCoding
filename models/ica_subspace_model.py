import numpy as np 
import tensorflow as tf 
#import utils.plot_functions as pf 
#import utils.data_processing as dp 
#from models.base_model import Model 

class IcaSubspaceModel(IcaModel):

  def __init__(self):
    super(IcaSubspaceModel, self).__init__()

  def load_params(self, params):
    """Parameters are the same as classical ICA, with addition of number of subspace/groups and
    group sizes for each subspace.
    
    Variables:
      num_groups: (int) number of groups/subspaces
      group_sizes: (int[]) number of vectors at each subspace. ith index for ith group.
    
    """
    super(IcaSubspaceModel, self).load_params(params)

    # new params for subspace ica
    self.num_groups = self.params.num_groups
    self.group_sizes = self.construct_group_sizes()
    

  def build_graph_from_input(self):
    """Build the Tensorflow graph object. 

    Placehodlers:
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
          Q, R = np.linalg.gr(np.random.standard_normal(self.w_analysis_shape))
          self.w_synth = tf.get_variable(name="w_synth", dtype=tf.float32),
            initializer=Q.astype(np.float32), trainable=True)
          self.w_analy = tf.tranpose(self.w_synth, name="w_analy")
          self.trainable_variables[self.w_synth.name] = self.w_synth

        with tf.name_scope("inference") as scope:
          self.s = tf.matmul(self.w_analy.T, self.input_img, name="latent_variables") 

        with tf.name_scope("output") as scope:
          with tf.name_scope("image_estimate"):
            self.recon = tf.matmul(self.w_synth, self.s, name="reconstruction")

    self.graph_built = True

  def compute_weight_gradient(self, optimizer, weight_op=None):
    return None
    
  def construct_group_sizes():
      """Construct respective group sizes. If group_size initialzed as None, then group sizes are uniformally
      distributed; unless specified otherwise. """
      if self.group_sizes is None:
          self.group_sizes = [self.num_neurons // self.num_groups for _ in self.num_groups]
      assert sum(self.group_sizes) == self.num_neurons, ("Total number of vectors should be the same "
                                                        "as number of neurons.")
      print("construct_group_sizes: {}".format(self.group_sizes))
      return self.group_sizes

  def get_subspace(g):
    """Return the vectors in the g-th subspace. """

    
