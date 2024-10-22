import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from mnist_dataset import NoisyMnistTwoView

import vcca

batch_size = 100

###########
# Dataset #
###########
# Loading the MNIST dataset
data_path = './noisy_mnist_two_views.pkl'
train_set = NoisyMnistTwoView(data_path, split='train')
test_set = NoisyMnistTwoView(data_path, split='valid')

# Initialization of the data loader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


##########
evaluate_every = 100000

writer = None
epochs = 100
checkpoint_every = 5
backup_every = 10000
model = vcca.VCCA(input_dims=[784, 784], latent_dim_shared=30, latent_dims_private=[30, 30],
                  output_activations=['sigmoid', 'sigmoid'],
                  recon_loss_types=['mse_fixed', 'mse_fixed']).cuda()
experiment_dir='./vcca_exp'

# checkpoint_path = 'exp/checkpoint_10.pt'
# model.load(checkpoint_path)

for epoch in tqdm(range(epochs)):
    for data in tqdm(train_loader):
        model.train_step(data)

    if epoch % checkpoint_every == 0:
        tqdm.write('Storing model checkpoint')
        model.save(os.path.join(experiment_dir, 'checkpoint_%02d.pt' % epoch))
