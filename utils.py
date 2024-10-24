import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus

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
