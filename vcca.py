import math
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer
import torch.optim as optimizer_module
# import gaussian_diffusion
# from nn import timestep_embedding
# from ddpm.denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch.distributions import Normal, Independent

def normal_kl(mu1, logvar1, mu2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mu1, logvar1, mu2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mu1 - mu2) ** 2) * torch.exp(-logvar2)
    )


def normal_dist(mu, logvar):
    # Return a factorized Normal distribution
    return Independent(Normal(loc=mu, scale=torch.exp(0.5 * logvar)), 1)


def compute_recon_loss(x_input, recon_mu, recon_logvar, loss_type, fixed_std=1.0, eps=1e-7):

    if loss_type == 'cross-entropy':
        # Smoothing of targets, which are assumed to be in [0, 1].
        x_input = x_input * 0.98 + 0.01
        loss = - x_input * torch.log(recon_mu + eps) + (1 - x_input) * torch.log(1 - recon_mu + eps)
    elif loss_type == 'mse_fixed':
        # Least squares loss, with specified std.
        loss = 0.5 * (((x_input - recon_mu) / fixed_std) ** 2)  # + 0.5 * math.log(2 * math.pi * fixed_std **2)
    elif loss_type == 'mse_learned':
        # Least squares loss, with learned std.
        loss = 0.5 * ((x_input - recon_mu) ** 2) / (torch.exp(recon_logvar) + eps) + 0.5 * recon_logvar  # + 0.5 * math.log(2 * math.pi)
    else:
        raise NotImplemented(f'Unsupported loss type: {loss_type}')

    return loss


# utility function to initialize an optimizer from its name
def init_optimizer(optimizer_name, params):
    assert hasattr(optimizer_module, optimizer_name)
    OptimizerClass = getattr(optimizer_module, optimizer_name)
    return OptimizerClass(params)


class DNN(nn.Module):

    def __init__(self, input_dim, output_dim, loss_types, hidden_dim=1024, dropout_rate=0.0, return_gaussian_dist=False):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_types = loss_types
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.return_gaussian_dist = return_gaussian_dist
        self._net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            # If returning a Gaussian distribution, we need both mean and log variance.
            nn.Linear(self.hidden_dim, self.output_dim * (1 + int(self.return_gaussian_dist))),
        )

    def forward(self, x):
        out = self._net(x)
        if not self.return_gaussian_dist:
            return out, None

        mu, logvar = out[:, :self.output_dim], out[:, self.output_dim:]
        return mu, logvar


class VCCA(nn.Module):

    def __init__(self, input_dims=[784, 784], latent_dim_shared=30, latent_dims_private=[30, 30],
                 recon_loss_types=['gaussian_unit_scale', 'gaussian_unit_scale'],
                 log_loss_every=10, writer=None, optimizer_name='Adam', lr=1e-4):
        super(VCCA, self).__init__()

        # TODO(weiranwang): parse args.
        # TODO(weiranwang): modify the backbone of networks to use UNET.

        self.input_dims = input_dims
        self.num_views = len(input_dims)
        self.latent_dim_shared = latent_dim_shared
        assert len(latent_dims_private) == self.num_views
        self.latent_dims_private = latent_dims_private
        self.encoders_shared = []
        self.encoders_private = []
        self.decoders = []
        for idim, hdim in zip(self.input_dims, self.latent_dims_private):
            self.encoders_shared.append(DNN(input_dim=idim, output_dim=self.latent_dim_shared, return_gaussian_dist=True))
            self.encoders_private.append(DNN(input_dim=idim, output_dim=hdim, return_gaussian_dist=True))
            self.decoders.append(DNN(input_dim=(self.latent_dim_shared + hdim), output_dim=idim, return_gaussian_dist=True))

        self.writer = writer
        self.log_loss_every = log_loss_every

        self.loss_items = {}
        self.opt = init_optimizer(optimizer_name, [
            # How to specify parameters for a list of modules.
            {'params': self.parameters(), 'lr': lr},
        ])
        self.iterations = 0

    def get_device(self):
        return list(self.parameters())[0].device

    def train_step(self, data):
        # Set all the models in training mode
        self.train(True)

        # Log the values in loss_items every log_loss_every iterations
        if self.writer is not None:
            if (self.iterations + 1) % self.log_loss_every == 0:
                self._log_loss()

        # Move the data to the appropriate device
        device = self.get_device()

        for i, item in enumerate(data):
            data[i] = item.to(device)

        # Perform the training step and update the iteration count
        self._train_step(data)
        self.iterations += 1

    def _add_loss_item(self, name, value):
        assert isinstance(name, str)
        assert isinstance(value, float) or isinstance(value, int)

        if not (name in self.loss_items):
            self.loss_items[name] = []

        self.loss_items[name].append(value)

    def _log_loss(self):
        # Log the expected value of the items in loss_items
        for key, values in self.loss_items.items():
            self.writer.add_scalar(tag=key, scalar_value=np.mean(values), global_step=self.iterations)
            self.loss_items[key] = []

    def save(self, model_path):
        items_to_save = self._get_items_to_store()
        items_to_save['iterations'] = self.iterations

        # Save the model and increment the checkpoint count
        torch.save(items_to_save, model_path)

    def load(self, model_path):
        items_to_load = torch.load(model_path)
        for key, value in items_to_load.items():
            assert hasattr(self, key)
            attribute = getattr(self, key)

            # Load the state dictionary for the stored modules and optimizers
            if isinstance(attribute, nn.Module) or isinstance(attribute, Optimizer):
                attribute.load_state_dict(value)

                # Move the optimizer parameters to the same correct device.
                # see https://github.com/pytorch/pytorch/issues/2830 for further details
                if isinstance(attribute, Optimizer):
                    device = list(value['state'].values())[0]['exp_avg'].device # Hack to identify the device
                    for state in attribute.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)

            # Otherwise just copy the value
            else:
                setattr(self, key, value)

    def _get_items_to_store(self):
        items_to_store = {}
        # store the network and optimizer parameters
        items_to_store['params'] = self.state_dict()
        items_to_store['opt'] = self.opt.state_dict()

        return items_to_store

    def _train_step(self, data):
        loss = self._compute_loss(data)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def _compute_loss(self, data):
        assert len(data) == self.num_views
        device = self.get_device()
        data = [x.to(device) for x in data]

        hs = []
        hp = []
        # prior_hs = []
        prior_hp = []
        latent_loss = 0.0
        for i, x in enumerate(data):

            enc_s = self.encoders_shared[i]
            sigma, logvar = enc_s(x)
            latent_kl = normal_kl(sigma, logvar, torch.zeros_like(sigma), torch.zeros_like(logvar)).sum(-1).mean()
            self._add_loss_item(f'latent_loss_shared_{i}', latent_kl)
            latent_loss += latent_kl
            latent_sample = normal_dist(sigma, logvar).sample()
            hs.append(latent_sample)
            # prior_hs.append(normal_dist(torch.zeros_like(sigma), torch.zeros_like(logvar)).sample())

            # If using latent diffusion, diffusion loss becomes latent loss, and diffusion sample becomes latent sample.
            enc_p = self.encoders_private[i]
            sigma, logvar = enc_p(x)
            latent_kl = normal_kl(sigma, logvar, torch.zeros_like(sigma), torch.zeros_like(logvar)).sum(-1).mean()
            self._add_loss_item(f'latent_loss_private_{i}', latent_kl)
            latent_loss += latent_kl
            latent_sample = normal_dist(sigma, logvar).sample()
            hp.append(latent_sample)
            prior_hp.append(normal_dist(torch.zeros_like(sigma), torch.zeros_like(logvar)).sample())

        # TODO(weiranwang): Configure how to aggregate the shared representations. Below we perform all-pairs recon.
        # TODO(weiranwang): Check the MMVAE without compromise paper for details.
        recon_loss = 0.0
        for i, x, dec, loss_type in enumerate(zip(data, self.decoders, self.loss_types)):
            for j in range(self.num_views):
                if j == i:
                    z = torch.cat([hs[j], prior_hp[i]], axis=1)
                else:
                    z = torch.cat([hs[j], hp[i]], axis=1)
                recon_mu, recon_logvar = dec(z)
                recon_loss_j_i = compute_recon_loss(x, recon_mu, recon_logvar, loss_type).sum(-1).mean()
                recon_loss += loss_j_i

        # after generating samples simply use mib loss?

        return loss  # losses['loss'].mean()

    def generate(self, shape):
        # return self.diffusion.p_sample_loop(shape)
        return self.diffusion.ddim_sample(shape)