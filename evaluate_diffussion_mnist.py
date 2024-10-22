
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import diffusion_model

batch_size = 200

###########
# Dataset #
###########
# Loading the MNIST dataset
mnist_dir = '/data/'
test_set = MNIST(mnist_dir, download=False, train=False, transform=ToTensor())

# Initialization of the data loader
test_set = DataLoader(test_set, batch_size=batch_size, shuffle=True)


##########

model = diffusion_model.DiffusionModel(
    input_dim=784, sampling_timesteps=50).cuda()
checkpoint_path = 'exp/checkpoint_95.pt'

model.load(checkpoint_path)
model.eval()

# import pdb;pdb.set_trace()
batch_size = 50
noise = torch.randn([batch_size, 1, 28, 28])
samples = model.generate(noise).cpu()
# samples[-1]

import matplotlib.pyplot as plt

for sample in samples:
    img = sample.reshape(28,28) # First image in the training set.
    plt.imshow(img, cmap='gray')
    plt.show() # Show the image


