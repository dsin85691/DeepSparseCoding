import torch
import torch.nn as nn
import torch.nn.functional as F

from DeepSparseCoding.modules.activations import lca_threshold
from DeepSparseCoding.utils.run_utils import compute_conv_output_shape
import DeepSparseCoding.utils.data_processing as dp


class LcaModule(nn.Module):
    """
    Keyword arguments:
        params: [dict] with keys:
            data_shape [list of int] of shape [elements, channels, height, width]; Assumes h = w (i.e. square inputs)
        The remaining keys are only used layer_types[0] is "conv":
            kernel_size: [int] edge size of the square convolving kernel
            stride: [int] vertical and horizontal stride of the convolution
            padding: [int] zero-padding added to both sides of the input
    TODO: Inference process should be streamlined by defining only a single step and iterating it in forward() as is done here:
    https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html

    TODO: Remove dependency on data_shape to make more intuitive in a hierarchy. i.e. use layer_channels as is done in the mlp
    """
    def setup_module(self, params):
        self.params = params
        if self.params.layer_types[0] == 'fc':
            self.layer_output_shapes = [[self.params.layer_channels]]
            self.w_shape = [self.params.num_pixels, self.params.layer_channels]
        else:
            self.layer_output_shapes = [self.params.data_shape] # [channels, height, width]
            assert (self.params.data_shape[-1] % self.params.stride == 0), (
              f'Stride = {self.params.stride} must divide evenly into input edge size = {self.params.data_shape[-1]}')
            self.w_shape = [
                self.params.layer_channels,
                self.params.data_shape[0], # channels = 1
                self.params.kernel_size,
                self.params.kernel_size
            ]
            output_height = compute_conv_output_shape(
                self.layer_output_shapes[-1][1],
                self.params.kernel_size,
                self.params.stride,
                self.params.padding,
                dilation=1)
            output_width = compute_conv_output_shape(
                self.layer_output_shapes[-1][2],
                self.params.kernel_size,
                self.params.stride,
                self.params.padding,
                dilation=1)
            self.layer_output_shapes.append([self.params.layer_channels, output_height, output_width])
        w_init = torch.randn(self.w_shape)
        w_init_normed = dp.l2_normalize_weights(w_init, eps=self.params.eps)
        self.weight = nn.Parameter(w_init_normed, requires_grad=True)

    def preprocess_data(self, input_tensor):
        if self.params.layer_types[0] == 'fc':
            input_tensor = input_tensor.view(self.params.batch_size, -1)
        return input_tensor

    def compute_excitatory_current(self, input_tensor, a_in):
        if self.params.layer_types[0] == 'fc':
            excitatory_current = torch.matmul(input_tensor, self.weight)
        else:
            recon = self.get_recon_from_latents(a_in)
            recon_error = input_tensor - recon
            error_injection = F.conv2d(
                input=recon_error,
                weight=self.weight,
                bias=None,
                stride=self.params.stride,
                padding=self.params.padding
            )
            excitatory_current = error_injection + a_in
        return excitatory_current

    def compute_inhibitory_connectivity(self):
        if self.params.layer_types[0] == 'fc':
            inhibitory_connectivity = torch.matmul(torch.transpose(self.weight, dim0=0, dim1=1),
                self.weight) - torch.eye(self.params.layer_channels,
                requires_grad=True, device=self.params.device)
        else:
            inhibitory_connectivity = 0 # TODO: return Grammian along channel dim for a single kernel location
        return inhibitory_connectivity

    def threshold_units(self, u_in):
        a_out = lca_threshold(u_in, self.params.thresh_type, self.params.rectify_a,
            self.params.sparse_mult)
        return a_out

    def step_inference(self, u_in, a_in, excitatory_current, inhibitory_connectivity, step):
        if self.params.layer_types[0] == 'fc':
            lca_explain_away = torch.matmul(a_in, inhibitory_connectivity)
        else:
            lca_explain_away = 0 # already computed in excitatory_current
        du = excitatory_current - lca_explain_away - u_in
        u_out = u_in + self.params.step_size * du
        return u_out, lca_explain_away

    def infer_coefficients(self, input_tensor):
        output_shape = [input_tensor.shape[0]] + self.layer_output_shapes[-1]
        u_list = [torch.zeros(output_shape, device=self.params.device)]
        a_list = [self.threshold_units(u_list[0])]
        excitatory_current = self.compute_excitatory_current(input_tensor, a_list[-1])
        inhibitory_connectivity = self.compute_inhibitory_connectivity()
        for step in range(self.params.num_steps-1):
            u = self.step_inference(
                u_list[step],
                a_list[step],
                excitatory_current,
                inhibitory_connectivity,
                step
            )[0]
            u_list.append(u)
            a_list.append(self.threshold_units(u))
            if self.params.layer_types[0] == 'conv':
                excitatory_current = self.compute_excitatory_current(input_tensor, a_list[-1])
        return (u_list, a_list)

    def get_recon_from_latents(self, a_in):
        if self.params.layer_types[0] == 'fc':
            recon = torch.matmul(a_in, torch.transpose(self.weight, dim0=0, dim1=1))
        else:
            recon = F.conv_transpose2d(
                input=a_in,
                weight=self.weight,
                bias=None,
                stride=self.params.stride,
                padding=self.params.padding
            )
        return recon

    def forward(self, input_tensor):
        u_list, a_list = self.infer_coefficients(input_tensor)
        return a_list[-1]

    def get_encodings(self, input_tensor):
        return self.forward(input_tensor)
