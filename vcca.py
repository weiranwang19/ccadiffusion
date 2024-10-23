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
from torch.nn.functional import softplus

"""
MVIB:

https://github.com/mfederici/Multi-View-Information-Bottleneck/tree/master
"""

# Auxiliary network for mutual information estimation
class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1


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
    if loss_type == 'cross_entropy':
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

    def __init__(self, input_dim, output_dim, output_activation, hidden_dim=1024, dropout_rate=0.0, return_gaussian_dist=True):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.return_gaussian_dist = return_gaussian_dist
        self._net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            # If returning a Gaussian distribution, we need both mean and log variance.
            nn.Linear(self.hidden_dim, self.output_dim * (1 + int(self.return_gaussian_dist))),
        )
        if output_activation is None:
            self.output_activation = nn.Identity()
        elif output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        else:
            raise ValueError(f'Unsupported output activation {output_activation}')

    def forward(self, x):
        out = self._net(x)
        if not self.return_gaussian_dist:
            return self.output_activation(out), None

        mu, logvar = out[:, :self.output_dim], out[:, self.output_dim:]
        # Only the mean receives activation.
        return self.output_activation(mu), logvar


class VCCA(nn.Module):

    def __init__(self, input_dims=[784, 784], latent_dim_shared=30, latent_dims_private=[30, 30],
                 output_activations=[None, None], recon_loss_types=['mse_fixed', 'mse_fixed'],
                 dropout_rate=0.0,
                 log_loss_every=10, writer=None, optimizer_name='Adam', lr=1e-4):
        super(VCCA, self).__init__()

        # TODO(weiranwang): parse args.
        # TODO(weiranwang): modify the backbone of networks to use UNET.

        self.input_dims = input_dims
        self.output_activations = output_activations
        self.recon_loss_types = recon_loss_types
        self.latent_dim_shared = latent_dim_shared
        self.latent_dims_private = latent_dims_private

        self.num_views = len(input_dims)
        assert self.num_views == len(self.output_activations)
        assert self.num_views == len(self.recon_loss_types)
        assert self.num_views == len(self.latent_dims_private)

        self.encoders_shared = nn.ModuleList()
        self.encoders_private = nn.ModuleList()

        self.decoders = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.hidden_dropout_layer = nn.Dropout(self.dropout_rate)

        for idim, hdim, act in zip(self.input_dims, self.latent_dims_private, self.output_activations):
            self.encoders_shared.append(
                DNN(input_dim=idim, output_dim=self.latent_dim_shared, dropout_rate=self.dropout_rate,
                    output_activation=None, return_gaussian_dist=True)
            )
            if hdim > 0:
                self.encoders_private.append(
                    DNN(input_dim=idim, output_dim=hdim, dropout_rate=self.dropout_rate,
                        output_activation=None, return_gaussian_dist=True)
                )
            self.decoders.append(
                DNN(input_dim=(self.latent_dim_shared + hdim), output_dim=idim, dropout_rate=self.dropout_rate,
                    output_activation=act, return_gaussian_dist=True)
            )
        # Initialization of the mutual information estimation network
        self.mi_estimator = MIEstimator(self.latent_dim_shared, self.latent_dim_shared)

        self.writer = writer
        self.log_loss_every = log_loss_every

        self.loss_items = {}
        self.opt = init_optimizer(optimizer_name, [
            # How to specify parameters for a list of modules.
            {'params': self.parameters(), 'lr': lr},
        ])
        self.iterations = 0

    def get_device(self):
        return list(self.encoders_shared[0].parameters())[0].device

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
        items_to_store['encoders_shared'] = self.encoders_shared.state_dict()
        items_to_store['encoders_private'] = self.encoders_private.state_dict()
        items_to_store['decoders'] = self.decoders.state_dict()
        items_to_store['mi_estimator'] = self.mi_estimator.state_dict()
        items_to_store['opt'] = self.opt.state_dict()
        return items_to_store

    def _train_step(self, data):
        loss = self._compute_loss(data)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def _compute_loss(self, data):
        assert len(data) == self.num_views

        hs = []
        hp = []
        # prior_hs = []
        dists = []
        prior_hp = []
        latent_loss = 0.0
        for i, (x, hdim) in enumerate(zip(data, self.latent_dims_private)):
            enc_s = self.encoders_shared[i]
            sigma, logvar = enc_s(x)
            latent_kl = normal_kl(sigma, logvar, torch.zeros_like(sigma), torch.zeros_like(logvar)).sum(-1).mean()
            self._add_loss_item(f'latent_loss_shared_{i}', float(latent_kl))
            latent_loss += latent_kl
            # Must use rsample to allow reparameterization.
            dist = normal_dist(sigma, logvar)
            dists.append(dist)
            latent_sample = dist.rsample()
            # import pdb;pdb.set_trace()
            hs.append(latent_sample)
            # prior_hs.append(normal_dist(torch.zeros_like(sigma), torch.zeros_like(logvar)).rsample())

            if hdim > 0:
                # If using latent diffusion, diffusion loss becomes latent loss, and diffusion sample becomes latent sample.
                enc_p = self.encoders_private[i]
                sigma, logvar = enc_p(x)
                latent_kl = normal_kl(sigma, logvar, torch.zeros_like(sigma), torch.zeros_like(logvar)).sum(-1).mean()
                self._add_loss_item(f'latent_loss_private_{i}', float(latent_kl))
                latent_loss += latent_kl
                latent_sample = normal_dist(sigma, logvar).rsample()
                hp.append(latent_sample)
                prior_hp.append(normal_dist(torch.zeros_like(sigma), torch.zeros_like(logvar)).rsample())

        # TODO(weiranwang): Configure how to aggregate the shared representations. Below we perform all-pairs recon.
        # TODO(weiranwang): Check the MMVAE without compromise paper for details.
        recon_loss = 0.0
        for i, (x, hdim, dec, loss_type) in enumerate(zip(data, self.latent_dims_private, self.decoders, self.recon_loss_types)):
            # import pdb;pdb.set_trace()
            for j in range(self.num_views):
                if hdim == 0:
                    z = hs[j]
                else:
                    if j == i:
                        z = torch.cat([hs[j], prior_hp[i]], axis=1)
                    else:
                        z = torch.cat([hs[j], hp[i]], axis=1)
                z = self.hidden_dropout_layer(z)
                recon_mu, recon_logvar = dec(z)
                recon_loss_j_i = compute_recon_loss(x, recon_mu, recon_logvar, loss_type).sum(-1).mean()
                self._add_loss_item(f'recon_loss_{j}_{i}', float(recon_loss_j_i))
                recon_loss += recon_loss_j_i

        # Sample from the posteriors with reparametrization
        # Encode a batch of data
        p_z1_given_v1 = dists[0]
        p_z2_given_v2 = dists[1]

        # Sample from the posteriors with reparametrization
        z1 = p_z1_given_v1.rsample()
        z2 = p_z2_given_v2.rsample()

        # Mutual information estimation
        mi_gradient, mi_estimation = self.mi_estimator(z1, z2)
        mi_gradient = mi_gradient.mean()
        mi_estimation = mi_estimation.mean()

        # Symmetrized Kullback-Leibler divergence
        kl_1_2 = p_z1_given_v1.log_prob(z1) - p_z2_given_v2.log_prob(z1)
        kl_2_1 = p_z2_given_v2.log_prob(z2) - p_z1_given_v1.log_prob(z2)
        skl = (kl_1_2 + kl_2_1).mean() / 2.

        # Update the value of beta according to the policy
        # beta = self.beta_scheduler(self.iterations)
        beta = 1.0

        # Logging the components
        self._add_loss_item('loss/I_z1_z2', mi_estimation.item())
        self._add_loss_item('loss/SKL_z1_z2', skl.item())
        self._add_loss_item('loss/beta', beta)

        # Computing the loss function
        mib_loss = - mi_gradient + beta * skl

        # after generating samples use mib loss?
        # assign weight and beta
        return latent_loss + recon_loss # + mib_loss

    def generate(self, shape):
        # return self.diffusion.p_sample_loop(shape)
        return self.diffusion.ddim_sample(shape)
