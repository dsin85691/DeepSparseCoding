import os
import types

import numpy as np
import torch
import torch.optim as optim

from DeepSparseCoding.params.base_params import BaseParams


class params(BaseParams):
    def set_params(self):
        super(params, self).set_params()
        self.model_type = 'mlp'
        self.model_name = 'mlp_768_cifar10'
        self.version = '0'
        self.dataset = 'cifar10'
        self.standardize_data = False
        self.rescale_data_to_one = False
        self.num_pixels = 32*32*3
        self.batch_size = 1
        self.num_epochs = 15
        self.weight_lr = 5e-4
        self.weight_decay = 2e-6
        self.layer_types = ['fc', 'fc']
        self.layer_channels = [self.num_pixels, 100, 10] # This was previously [784, 768, 10]
        self.activation_functions = ['lrelu', 'identity']
        self.dropout_rate = [0.5, 0.0] # probability of value being set to zero
        self.train_logs_per_epoch = 4
        self.optimizer = types.SimpleNamespace()
        self.optimizer.name = 'adam'
        self.optimizer.lr_annealing_milestone_frac = [0.8] # fraction of num_epochs
        self.optimizer.lr_decay_rate = 0.1

    def compute_helper_params(self):
        super(params, self).compute_helper_params()
        self.optimizer.milestones = [frac * self.num_epochs
            for frac in self.optimizer.lr_annealing_milestone_frac]
