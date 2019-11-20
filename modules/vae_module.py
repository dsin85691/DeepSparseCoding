import numpy as np
import tensorflow as tf
from ops.init_ops import L2NormalizedTruncatedNormalInitializer
import utils.entropy_functions as ef
from utils.trainable_variable_dict import TrainableVariableDict
from modules.ae_module import AeModule

class VaeModule(AeModule):
  def __init__(self, data_tensor, layer_types, output_channels, patch_size, conv_strides,
    decay_mult, norm_mult, kld_mult, act_funcs, dropout, tie_decoder_weights, noise_level,
    recon_loss_type, norm_w_init, variable_scope="vae"):
    """
    Variational Autoencoder module
    Inputs:
      data_tensor
      output_channels [list of ints] A list of channels to make, also defines number of layers
      decay_mult [float] tradeoff multiplier for weight decay loss
      norm_mult [float] tradeoff multiplier for weight norm loss (asks weight norm to == 1)
      kld_mult [float] tradeoff multiplier for latent variational kld loss
      act_funcs [list of functions] activation functions
      dropout [list of floats] specifies the keep probability or None
      noise_level [float] stddev of noise to be added to the input (for denoising VAE)
      recon_loss_type [str] either "mse" or the cross entropy loss used in Kingma & Welling
      conv_strides [list] list of strides for convolution [batch, y, x, channels]
      patch_size: number of (y, x) inputs for convolutional patches
      norm_w_init: if True, l2 normalize w_init,
        reducing over [0] axis on enc and [-1] axis on dec
      variable_scope [str] specifies the variable_scope for the module
    Outputs:
      dictionary
    """
    self.noise_level = noise_level
    with tf.compat.v1.variable_scope(variable_scope) as scope:
      if self.noise_level > 0.0:
          self.corrupt_data = tf.add(tf.random.normal(shape=tf.shape(data_tensor),
            mean=tf.reduce_mean(data_tensor), stddev=noise_level, dtype=tf.float32),
            data_tensor, name="noisy_data")
      else:
        self.corrupt_data = tf.identity(data_tensor, name="clean_data")
    self.recon_loss_type = recon_loss_type
    self.kld_mult = kld_mult
    super(VaeModule, self).__init__(data_tensor, layer_types, output_channels, patch_size,
      conv_strides, decay_mult, norm_mult, act_funcs, dropout, tie_decoder_weights, norm_w_init,
      variable_scope)

  def compute_recon_loss(self, reconstruction):
    if self.recon_loss_type == "mse":
      return super(VaeModule, self).compute_recon_loss(reconstruction)
    elif self.recon_loss_type == "crossentropy":
      reduc_dim = list(range(1, len(reconstruction.shape)))# We want to avg over batch
      #recon_loss = tf.reduce_mean(-tf.reduce_sum(self.data_tensor * ef.safe_log(reconstruction) \
      #  + (1-self.data_tensor) * ef.safe_log(1-reconstruction), axis=reduc_dim))
      #return recon_loss
      recon_loss = tf.reduce_mean(-tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits_v2(reconstruction, self.data_tensor),
        axis=[-1]))#reduc_dim))
      return recon_loss
    else:
      assert False, ("recon_loss_type param must be `mse` or `crossentropy`")

  def log_normal_pdf(self, sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    lnp = tf.reduce_sum(
      -0.5 * (tf.square(sample - mean) * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)
    return lnp

  def compute_latent_loss(self, act, mean, logvar):
    with tf.compat.v1.variable_scope("latent"):
      reduc_dim = list(range(1, len(mean.shape))) # Want to avg over batch, sum over the rest
      latent_loss = self.kld_mult * tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + 2.0 * logvar
        - tf.square(mean) - tf.exp(2.0 * logvar), reduc_dim))
      #logpz = self.log_normal_pdf(act, 0., 0., reduc_dim)
      #logqz_x = self.log_normal_pdf(act, mean, logvar, reduc_dim)
      #latent_loss = self.kld_mult * tf.reduce_mean(logpz - logqz_x)
    return latent_loss

  def compute_total_loss(self):
    with tf.compat.v1.variable_scope("loss") as scope:
      self.loss_dict = {"recon_loss":self.compute_recon_loss(self.reconstruction),
        "weight_decay_loss":self.compute_weight_decay_loss(),
        "weight_norm_loss":self.compute_weight_norm_loss(),
        "latent_loss":self.compute_latent_loss(self.act, self.latent_mean, self.latent_logvar)}
      self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")

  def reparameterize(self, mean, logvar, noise):
    #act = mean + logvar * noise
    #act = mean + tf.sqrt(tf.exp(logvar)) * noise
    #act = mean + tf.exp(logvar) * noise
    act = mean + tf.exp(0.5 * logvar) * noise
    return act

  def gen_noise(self, noise_type, shape):
    if noise_type == "standard_normal":
      return tf.random.normal(shape)
    assert False, ("Noise type "+noise_type+" not supported")

  def build_graph(self):
    with tf.compat.v1.variable_scope(self.variable_scope) as scope:
      with tf.compat.v1.variable_scope("weight_inits") as scope:
        self.w_init = tf.initializers.truncated_normal(mean=0.0, stddev=0.01)
        self.w_normal_init = tf.contrib.layers.variance_scaling_initializer()
        self.w_xavier_init = tf.contrib.layers.xavier_initializer(uniform=False)
        self.w_normed_enc_init = L2NormalizedTruncatedNormalInitializer(mean=0.0, stddev=0.001,
          axis=0, epsilon=1e-12, dtype=tf.float32) #TODO: Fix axis to be general to conv layers
        self.w_normed_dec_init = L2NormalizedTruncatedNormalInitializer(mean=0.0, stddev=0.001,
          axis=-1, epsilon=1e-12, dtype=tf.float32)
        self.b_init = tf.initializers.constant(1e-5)

      self.u_list = [self.corrupt_data]
      self.w_list = []
      self.b_list = []
      enc_u_list, enc_w_list, enc_b_list = self.build_encoder(self.u_list[0],
        self.act_funcs[:self.num_encoder_layers])
      self.enc_u_list = enc_u_list # TODO: use u_list instead of enc_u_list in ae_module
      self.enc_w_list = enc_w_list
      self.enc_b_list = enc_b_list
      self.u_list += enc_u_list[:-1] # don't store the mean value in u_list
      self.w_list += enc_w_list
      self.b_list += enc_b_list

      # Add variance computation from encoder
      w_shape = self.w_list[-1].get_shape().as_list() # same shape as mean weights
      b_shape = self.b_list[-1].get_shape().as_list() # same shape as mean bias
      self.w_enc_std = tf.compat.v1.get_variable(name="w_enc_"+str(self.num_encoder_layers)+"_std",
        shape=w_shape, dtype=tf.float32,
        initializer=self.w_xavier_init, trainable=True)
        #initializer=self.w_init, trainable=True)
      #if self.layer_types[-1] == "conv":
      #  b_shape = w_shape[-2]
      #else:
      #  b_shape = w_shape[-1]
      self.b_enc_std = tf.compat.v1.get_variable(name="b_enc_"+str(self.num_encoder_layers)+"_std",
        shape=b_shape, dtype=tf.float32, initializer=self.b_init, trainable=True)
      self.trainable_variables[self.w_enc_std.name] = self.w_enc_std
      self.trainable_variables[self.b_enc_std.name] = self.b_enc_std

      self.latent_mean = self.enc_u_list[-1]
      self.a = self.latent_mean # alias for AE model

      if self.layer_types[-1] == "conv":
        self.latent_logvar = tf.add(tf.nn.conv2d(self.u_list[-1], self.w_enc_std,
          self.conv_strides[self.num_encoder_layers-1], padding="SAME"), self.b_enc_std)
      else:
        #self.latent_logvar = 1e-8 + tf.nn.softplus(tf.matmul(self.u_list[-1],
        #  self.w_enc_std) + self.b_enc_std) # std must be positive
        self.latent_logvar = tf.add(tf.matmul(self.u_list[-1], self.w_enc_std),
          self.b_enc_std)

      noise = self.gen_noise(noise_type="standard_normal", shape=tf.shape(self.latent_logvar))
      self.act = tf.identity(self.reparameterize(self.latent_mean,
        self.latent_logvar, noise), name="activity")
      self.u_list.append(self.act)

      dec_u_list, dec_w_list, dec_b_list = self.build_decoder(self.u_list[-1],
        self.act_funcs[self.num_encoder_layers:])
      self.u_list += dec_u_list
      self.w_list += dec_w_list
      self.b_list += dec_b_list

      for w,b in zip(self.w_list, self.b_list):
        self.trainable_variables[w.name] = w
        self.trainable_variables[b.name] = b

      with tf.compat.v1.variable_scope("norm_weights") as scope:
        w_enc_norm_dim = list(range(len(self.w_list[0].get_shape().as_list())-1))
        self.norm_enc_w = self.w_list[0].assign(tf.nn.l2_normalize(self.w_list[0],
          axis=w_enc_norm_dim, epsilon=1e-8, name="row_l2_norm"))
        self.norm_dec_w = self.w_list[-1].assign(tf.nn.l2_normalize(self.w_list[-1],
          axis=-1, epsilon=1e-8, name="col_l2_norm"))
        self.norm_w = tf.group(self.norm_enc_w, self.norm_dec_w, name="l2_norm_weights")

      with tf.compat.v1.variable_scope("output") as scope:
        self.reconstruction = tf.identity(self.u_list[-1], name="reconstruction")

      self.compute_total_loss()
