import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
from  models.base_model import Model


class IcaSubspaceModel(Model):

    def __init__(self):
        super(IcaSubspaceModel, self).__init__()
        self.vector_inputs = True

    def load_params(self, params):
        super(IcaSubspaceModel, self).load_params(params)
        self.input_shape = [None, self.params.num_pixels]
        self.w_synth_shape = [self.params.num_pixels, self.params.num_neurons]
        self.w_analy_shape = [self.params.num_neurons, self.params.num_pixels]
        self.R = self.construct_index_matrix().astype(np.float32)

    def construct_index_matrix(self):
        R = np.zeros(shape=(self.params.num_neurons, self.params.num_groups))
        for g in range(self.params.num_groups):
            R[g:(g+1)*self.params.group_size, g] = 1
        return R

    def get_input_shape(self):
        return self.input_shape

    def build_graph_from_input(self, input_node):
        with tf.device(self.params.device):
            with self.graph.as_default():
                with tf.variable_scope("weights") as scope:
                    Q, _ = np.linalg.qr(np.random.standard_normal(self.w_analy_shape), mode='complete')
                    self.w_analy = tf.get_variable(name="w_analy",
                                                   dtype=tf.float32,
                                                   initializer=Q.astype(np.float32),
                                                   trainable=True)
                    self.w_synth = tf.transpose(self.w_analy, name="w_synth")
                    self.trainable_variables[self.w_analy.name] = self.w_analy

                with tf.variable_scope("inference") as scope:
                    self.s = tf.matmul(input_node, tf.transpose(self.w_analy), name="latent_vars")

#                with tf.variable_scope("log_liklihood") as scope:
#                    self.log_lik = self.compute_log_lik(input_node) 

                with tf.variable_scope("output") as scope:
                    self.recon = tf.matmul(self.s, tf.transpose(self.w_synth), name="recon")

                with tf.variable_scope("orthonormalize") as scope:
                    self.orthonorm_weights = tf.assign(self.w_analy, 
                            tf.matmul(tf.math.pow(tf.matmul(self.w_analy, tf.transpose(self.w_analy)), -0.5), 
                                self.w_analy))

        self.graph_built = True

    def compute_weight_gradients(self, optimizer, weight_op=None):
        W = weight_op
        grads = []
        for img_i in range(self.params.batch_size):
            I = tf.slice(self.input_placeholder, [img_i, 0], [1, self.params.num_pixels])
            I = tf.transpose(I)
            one_w_grad = self.compute_weight_gradient_per_input(I)
            self.one_grad = one_w_grad
            grads.append(one_w_grad)
        gradient = tf.stack(grads)
        avg_grad = tf.math.reduce_mean(gradient, axis=0)
        self.w_grad = avg_grad
        return [(avg_grad, weight_op)]
    
    def compute_weight_gradient_per_input(self, I):
        def nonlinearity(u, alpha=1):
            return -0.5 * alpha * tf.math.pow(u, -0.5)
        
        Wt_I = tf.matmul(tf.transpose(self.w_analy), I)
#        print("Wt_I shape", Wt_I.shape)
        Wt_I_sq = tf.math.pow(Wt_I, 2)
#        print("Wt_I_sq", Wt_I_sq.shape)
        pre_nonlinear_term = nonlinearity(tf.matmul(tf.transpose(Wt_I_sq), self.R))
        nonlinear_term = tf.matmul(pre_nonlinear_term, tf.transpose(self.R))

#        print("nonlinear term shape", nonlinear_term.shape)
        repeat_I = tf.tile(I, [1, self.params.num_neurons])

        self.repeat_I = repeat_I
        self.wti = Wt_I
        self.pre_nonlinear = pre_nonlinear_term
        self.nonlinear = nonlinear_term
#        print("repeat I", repeat_I.shape)
        return  repeat_I * tf.transpose(Wt_I) * nonlinear_term

        
    def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
        update_dict = super(IcaSubspaceModel, self).generate_update_dict(input_data, input_labels, batch_step)
        feed_dict = self.get_feed_dict(input_data, input_labels)
        eval_list  = [self.global_step, self.s, self.recon, self.w_analy, self.w_grad, self.one_grad, self.repeat_I, self.wti, self.nonlinear, self.pre_nonlinear]
        out_vals = tf.get_default_session().run(eval_list, feed_dict)
        print("w_analy")
        print(out_vals[3])
        print("w_grad")
        print(out_vals[4])
        print("one_grad")
        print(out_vals[5])
        print("repeat I")
        print(out_vals[6])
        print("wtf")
        print(out_vals[7])
        print("nonlinear")
        print(out_vals[8])
        print("prenonlinear")
        print(out_vals[9])
        stat_dict = {
                "global_step": out_vals[0],
                "latent_vars": out_vals[1],
                "recon": out_vals[2],
                "w_analy": out_vals[3],
                "w_grad": out_vals[4],
                }


        update_dict.update(stat_dict) # stat_dict vals overwrite
        return update_dict

    def generate_plots(self, input_data, input_labels=None):
        super(IcaSubspaceModel, self).generate_plots(input_data, input_labels)
    
        ## ADD FUCNITONS
        feed_dict = self.get_feed_dict(input_data, input_labels)
        eval_list = [self.global_step, self.w_synth, self.w_analy, self.s, self.w_grad]
        eval_out = tf.get_default_session().run(eval_list, feed_dict)
 
        # step
        curr_step = str(eval_out[0])
        w_shape = [self.params.num_neurons, self.params.patch_edge_size, self.params.patch_edge_size, 1]
        w_synth_eval = eval_out[1].reshape(w_shape)
        w_analy_eval = eval_out[2].reshape(w_shape)
        w_grad_eval = eval_out[4].reshape(w_shape)
        latent_vars = eval_out[3]
        pf.plot_weights(w_synth_eval, title="w_synth at step {}".format(curr_step), figsize=(16, 16),
                    save_filename="{}w_synth_eval_{}.png".format(self.params.disp_dir, curr_step))
        pf.plot_weights(w_analy_eval, title="w_synth at step {}".format(curr_step), figsize=(16, 16),
                    save_filename="{}w_analy_eval_{}.png".format(self.params.disp_dir, curr_step))
        pf.plot_weights(w_synth_eval.transpose(0, 2, 1, 3), title="w_synth at step {}".format(curr_step), figsize=(16, 16),
                    save_filename="{}w_synth_T_eval_{}.png".format(self.params.disp_dir, curr_step))
        pf.plot_weights(w_analy_eval.transpose(0, 2, 1, 3), title="w_synth at step {}".format(curr_step), figsize=(16, 16),
                    save_filename="{}w_analy_T_eval_{}.png".format(self.params.disp_dir, curr_step))
        pf.plot_weights(w_grad_eval, title="w_grad at step {}".format(curr_step), figsize=(16, 16),
                    save_filename="{}w_grad_{}.png".format(self.params.disp_dir, curr_step))
 
