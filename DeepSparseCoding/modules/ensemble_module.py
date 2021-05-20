import torch.nn as nn

import DeepSparseCoding.utils.loaders as loaders


class EnsembleModule(nn.Sequential):
    def __init__(self, params, logger=None):  # do not do Sequential's init
        super(nn.Sequential, self).__init__()
        self.params = params

    def setup_ensemble_module(self):
        for subparams in self.params.ensemble_params:
            submodule = loaders.load_module(subparams.model_type, subparams)
            #submodule.setup_module(subparams)
            self.add_module(subparams.model_type, submodule)
            if subparams.model_type == 'lca':
                self.register_parameter('w', submodule.getW())
            elif subparams.model_type == 'mlp':
                layers = submodule.getLayers()

                self.register_parameter('fc0_w', layers[0].weight)
                self.register_parameter('fc0_b', layers[0].bias)

    def forward(self, x):
        self.layer_list = [x]
        print(self.layer_list)
        for module in self:
            self.layer_list.append(module.get_encodings(self.layer_list[-1]))  # latent encodings
        return self.layer_list[-1]

    def get_encodings(self, x):
        return self.forward(x)
