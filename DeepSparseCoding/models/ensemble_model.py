import torch
import torch.nn as nn
import torchvision

import DeepSparseCoding.utils.loaders as loaders
from DeepSparseCoding.models.base_model import BaseModel
from DeepSparseCoding.modules.ensemble_module import EnsembleModule


class EnsembleModel(BaseModel, EnsembleModule):
    def __init__(self, params, logger=None):
        """
        Setup required model components
        """

        super(EnsembleModel, self).setup(params, logger)
        super(EnsembleModel, self).__init__(params, logger)
        self.setup_module(params)
        self.setup_optimizer()

    def setup_module(self, params):
        for subparams in params.ensemble_params:
            subparams.epoch_size = params.epoch_size
            subparams.batches_per_epoch = params.batches_per_epoch
            subparams.num_batches = params.num_batches
            # subparams.num_val_images = params.num_val_images
            # subparams.num_test_images = params.num_test_images
            subparams.data_shape = params.data_shape
        super(EnsembleModel, self).setup_ensemble_module()

        self.submodel_classes = []
        for submodel_params in self.params.ensemble_params:
            self.submodel_classes.append(loaders.load_model_class(submodel_params.model_type))

    def setup_optimizer(self):
        for module in self:
            module.optimizer = self.get_optimizer(
                optimizer_params=module.params,
                trainable_variables=module.parameters())
            module.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                module.optimizer,
                milestones=module.params.optimizer.milestones,
                gamma=module.params.optimizer.lr_decay_rate)

    def preprocess_data(self, data):
        """
        We assume that only the first submodel will be preprocessing the input data
        """
        submodule = self.__getitem__(0)
        return self.submodel_classes[0].preprocess_data(submodule, data)

    def get_total_loss(self, input_tuple, ensemble_index):
        submodule = self.__getitem__(ensemble_index)
        submodel_class = self.submodel_classes[ensemble_index]
        return submodel_class.get_total_loss(submodule, input_tuple)

    def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
        update_dict = super(EnsembleModel, self).generate_update_dict(input_data,
                                                                      input_labels, batch_step)
        x = input_data.clone()  # TODO: Do I need to clone it? If not then don't.
        for ensemble_index, submodel_class in enumerate(self.submodel_classes):
            submodule = self.__getitem__(ensemble_index)
            submodel_update_dict = submodel_class.generate_update_dict(submodule, x,
                                                                       input_labels, batch_step, update_dict=dict())
            for key, value in submodel_update_dict.items():
                if key not in ['epoch', 'batch_step']:
                    key = submodule.params.model_type + '_' + key
                update_dict[key] = value
            x = submodule.get_encodings(x)
        return update_dict


class NetTensorFlowWrapper(nn.Sequential):
    def __init__(self, main_module: nn.Sequential):
        super(NetTensorFlowWrapper, self).__init__()
        self.main_module = main_module
        self.params = self.main_module.params
        #self.setup_module()

    def get_module(self):
        return self.main_module

    def forward(self, x):
        x = torch.reshape(x, (100, 784))
        output_tensor = self.main_module(x)
        output_tensor = torch.squeeze(output_tensor, 1)
        output_tensor = torch.squeeze(output_tensor, 1)
        print(output_tensor.shape)
        return output_tensor
