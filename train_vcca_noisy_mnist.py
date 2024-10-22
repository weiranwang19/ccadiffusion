import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import vcca
from mnist_dataset import NoisyMnistTwoView

from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

from utils import EmbeddedDataset, build_matrix

data_path = './noisy_mnist_two_views.pkl'
experiment_dir='./vcca_exp'


def evaluate(encoder, encoder_name):
    device = encoder.device
    import matplotlib.pyplot as plt

    # Definition of scaler and Logistic classifier used to evaluate the different representations
    print('-Embedding the dataset')

    valid_set = NoisyMnistTwoView(data_path, split='valid', mode='view1')
    test_set = NoisyMnistTwoView(data_path, split='test', mode='view1')

    # Embed train and test set using the learned encoder
    embedded_train_set = EmbeddedDataset(base_dataset=valid_set, encoder=encoder, device=device)
    embedded_test_set = EmbeddedDataset(base_dataset=test_set, encoder=encoder, device=device)

    # Convert the two sets into 2D matrices for evaluation
    FX_train, Y_train = build_matrix(embedded_train_set)
    FX_test, Y_test = build_matrix(embedded_test_set)

    print('-Computing classifier accuracy')
    classifier = LogisticRegression(solver='saga', multi_class='multinomial', C=10, tol=.1)

    # Fit the linear classifier on the selection
    classifier.fit(FX_train, Y_train)
    # Evaluate the classifier on the embedded test set
    test_accuracy = classifier.score(FX_test, Y_test)
    print(f'Test Accuracy: {test_accuracy}')

    # Project the test set on the principal components
    tsne = TSNE(n_components=2, perplexity=20.0)
    projected_X_test = tsne.fit_transform(FX_test)

    # And plot the representation with different colors corresponding to the different labels
    plt.title(f'{encoder_name}', size=15)
    for label in range(10):
        selected_FX_test = projected_X_test[Y_test == label]
        plt.plot(selected_FX_test[:, 0], selected_FX_test[:, 1], 'o', label=label, alpha=0.05)



###########
# Dataset #
###########
# Loading the MNIST dataset
train_set = NoisyMnistTwoView(data_path, split='train')
test_set = NoisyMnistTwoView(data_path, split='valid')

# Initialization of the data loader
batch_size = 100
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

##########
evaluate_every = 100000

writer = None
epochs = 5
checkpoint_every = 1
writer = SummaryWriter(log_dir=experiment_dir)

model = vcca.VCCA(input_dims=[784, 784], latent_dim_shared=30, latent_dims_private=[30, 30],
                  output_activations=['sigmoid', 'sigmoid'],
                  recon_loss_types=['mse_fixed', 'mse_fixed'],
                  dropout_rate=0.2, writer=writer)
if torch.cuda.is_available():
    model = model.cuda()


for epoch in tqdm(range(epochs)):
    for data in tqdm(train_loader):
        model.train_step(data)

    if epoch % checkpoint_every == 0:
        tqdm.write('Storing model checkpoint')
        model.save(os.path.join(experiment_dir, 'checkpoint_%02d.pt' % epoch))

checkpoint_path = experiment_dir + '/checkpoint_40.pt'
model.load(checkpoint_path)
evaluate(model.encoders_shared[0], 'shared')
