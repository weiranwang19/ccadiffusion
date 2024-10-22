import os
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import diffusion_model

batch_size = 100

###########
# Dataset #
###########
# Loading the MNIST dataset
mnist_dir = '/data/'
mnist_dir = '~'
train_set = MNIST(mnist_dir, download=False, train=True, transform=ToTensor())
test_set = MNIST(mnist_dir, download=False, train=False, transform=ToTensor())

# Initialization of the data loader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


##########
evaluate_every = 100000

writer = None
epochs = 100
checkpoint_every = 5
backup_every = 10000
model = diffusion_model.DiffusionModel(input_dim=784) # .cuda()
experiment_dir='./exp'

# checkpoint_path = 'exp/checkpoint_10.pt'
# model.load(checkpoint_path)

for epoch in tqdm(range(epochs)):
    for data in tqdm(train_loader):
        model.train_step(data)

    if epoch % checkpoint_every == 0:
        tqdm.write('Storing model checkpoint')
        model.save(os.path.join(experiment_dir, 'checkpoint_%02d.pt' % epoch))

