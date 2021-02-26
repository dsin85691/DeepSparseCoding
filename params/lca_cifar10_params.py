import types

from DeepSparseCoding.params.base_params import BaseParams


class params(BaseParams):
    def set_params(self):
        super(params, self).set_params()
        self.model_type = 'lca'
        self.model_name = 'lca_cifar10'
        self.version = '0'
        self.dataset = 'cifar10'
        self.layer_types = ['conv']
        self.num_validation = 10000
        self.standardize_data = True
        self.rescale_data_to_one = False
        self.center_dataset = False
        self.batch_size = 25
        self.num_epochs = 500
        self.train_logs_per_epoch = 6
        self.renormalize_weights = True
        self.layer_channels = 128
        self.kernel_size = 8
        self.stride = 2
        self.padding = 0
        self.weight_decay = 0.0
        self.weight_lr = 0.001
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'sgd'
        self.optimizer.lr_annealing_milestone_frac = [0.8] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.8
        self.dt = 0.001
        self.tau = 0.1#0.2
        self.num_steps = 37#75
        self.rectify_a = True
        self.thresh_type = 'hard'
        self.sparse_mult = 0.35#0.30
        self.compute_helper_params()

    def compute_helper_params(self):
        super(params, self).compute_helper_params()
        self.optimizer.milestones = [frac * self.num_epochs
            for frac in self.optimizer.lr_annealing_milestone_frac]
        self.step_size = self.dt / self.tau
        self.out_channels = self.layer_channels
        self.num_pixels = 3072
        self.in_channels = 3
