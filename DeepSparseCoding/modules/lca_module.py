import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from DeepSparseCoding.modules.activations import lca_threshold


class LcaModule(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.w = nn.Parameter(
            F.normalize(
                torch.randn(self.params.num_pixels, self.params.num_latent),
                p=2, dim=0),
            requires_grad=True)

    def preprocess_data(self, input_tensor):
        if input_tensor.shape[0] == 100:
                input_tensor = torch.reshape(input_tensor, (100, 784))
        else:
                input_tensor = torch.reshape(input_tensor, (1, 784))
        return input_tensor

    def compute_excitatory_current(self, input_tensor):
        return torch.matmul(input_tensor, self.w)

    def compute_inhibitory_connectivity(self):
        lca_g = torch.matmul(torch.transpose(self.w, dim0=0, dim1=1),
            self.w) - torch.eye(self.params.num_latent,
            requires_grad=True, device=self.params.device)
        return lca_g

    def threshold_units(self, u_in):
        a_out = lca_threshold(u_in, self.params.thresh_type, self.params.rectify_a,
            self.params.sparse_mult)
        return a_out

    def step_inference(self, u_in, a_in, b, g, step):
        lca_explain_away = torch.matmul(a_in, g)
        du = b - lca_explain_away - u_in
        u_out = u_in + self.params.step_size * du
        return u_out, lca_explain_away

    def infer_coefficients(self, input_tensor):
        lca_b = self.compute_excitatory_current(input_tensor)
        lca_g = self.compute_inhibitory_connectivity()
        u_list = [torch.zeros([input_tensor.shape[0], self.params.num_latent],
            device=self.params.device)]
        a_list = [self.threshold_units(u_list[0])]
        # TODO: look into redoing this with a register_buffer that gets updated? look up simple RNN code...
        for step in range(self.params.num_steps-1):
            u = self.step_inference(u_list[step], a_list[step], lca_b, lca_g, step)[0]
            u_list.append(u)
            a_list.append(self.threshold_units(u))
        return (u_list, a_list)

    def get_recon_from_latents(self, latents):
        return torch.matmul(latents, torch.transpose(self.w, dim0=0, dim1=1))

    def get_encodings(self, input_tensor):
        u_list, a_list = self.infer_coefficients(input_tensor)
        return a_list[-1]

    def forward(self, input_tensor):
        latents = self.get_encodings(input_tensor)
        reconstruction = self.get_recon_from_latents(latents)
        return reconstruction

    def getW(self):
        return self.w

