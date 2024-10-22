import math
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer
import torch.optim as optimizer_module
# import gaussian_diffusion
# from nn import timestep_embedding
from ddpm.denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion


# utility function to initialize an optimizer from its name
def init_optimizer(optimizer_name, params):
    assert hasattr(optimizer_module, optimizer_name)
    OptimizerClass = getattr(optimizer_module, optimizer_name)
    return OptimizerClass(params)


class NetworkWithTimeInput(nn.Module):

    def __init__(self, input_dim, hidden_dim=1024, dropout_rate=0.0):
        super(NetworkWithTimeInput, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self._input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self._time_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self._network = nn.Sequential(
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
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, x, t):
        x_proj = self._input_proj(x) + self._time_proj(timestep_embedding(t, self.hidden_dim))
        return self._network(x_proj)


class DiffusionModel(nn.Module):

    def __init__(self, input_dim=784, timesteps=200, sampling_timesteps=100, log_loss_every=10, writer=None, optimizer_name='Adam', lr=1e-4):
        super(DiffusionModel, self).__init__()

        self.input_dim = input_dim
        # self.network = NetworkWithTimeInput(input_dim=input_dim)
        self.network = Unet(channels=1, dim=128, dim_mults = (1, 2, 4))
        self.diffusion = GaussianDiffusion(image_size=[28, 28], timesteps=timesteps, sampling_timesteps=sampling_timesteps, model=self.network)
        self.writer = writer
        self.log_loss_every = log_loss_every

        self.loss_items = {}
        self.opt = init_optimizer(optimizer_name, [
            {'params': self.network.parameters(), 'lr': lr},
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
        items_to_store['network'] = self.network.state_dict()
        items_to_store['opt'] = self.opt.state_dict()

        return items_to_store

    def _train_step(self, data):
        loss = self._compute_loss(data)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def _compute_loss(self, data):
        # import pdb;pdb.set_trace()
        x = data[0]
        batch_size = x.shape[0]
        x = torch.reshape(x, [batch_size, self.diffusion.channels] + self.diffusion.image_size)
        # num_diffusion_steps = self.diffusion.num_timesteps
        # t = torch.randint(low=0, high=num_diffusion_steps, size=[batch_size]).to(x.device)
        # # losses = self.diffusion(self.network, x, t)
        loss = self.diffusion(x)
        print(loss)
        # Add other terms.
        return loss  # losses['loss'].mean()

    def generate(self, shape):
        # return self.diffusion.p_sample_loop(shape)
        return self.diffusion.ddim_sample(shape)
