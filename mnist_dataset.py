import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import pickle


class NoisyMnistTwoViews(Dataset):
    def __init__(self, data_path, split, mode='unsup'):

        with open(data_path,'rb') as f:
            noisy_mnist = pickle.load(f)

        if split == 'train':
            X1, X2, Y, N = noisy_mnist['train_x1'], noisy_mnist['train_x2'], noisy_mnist['train_labels'], noisy_mnist['N_train']
        elif split == 'valid':
            X1, X2, Y, N = noisy_mnist['valid_x1'], noisy_mnist['valid_x2'], noisy_mnist['valid_labels'], noisy_mnist['N_valid']
        elif split == 'test':
            X1, X2, Y, N = noisy_mnist['test_x1'], noisy_mnist['test_x2'], noisy_mnist['test_labels'], noisy_mnist['N_test']
        else:
            raise ValueError(f'Unsupported split {split}')

        self.num_samples = N
        self.X1 = torch.from_numpy(X1).to(torch.float32)
        self.X2 = torch.from_numpy(X2).to(torch.float32)
        self.Y  = torch.from_numpy(Y)
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'unsup':
            return self.X1[index], self.X2[index]
        elif self.mode == 'view1':
            return self.X1[index], self.Y[index]
        elif self.mode == 'view2':
            return self.X2[index], self.Y[index]
        else:
            raise ValueError(f'Unsupported mode {self.mode}')

    def __len__(self):
        return self.num_samples


class PixelCorruption(object):
    MODALITIES = ['flip', 'drop']

    def __init__(self, p, min=0, max=1, mode='drop'):
        super(PixelCorruption, self).__init__()

        assert mode in self.MODALITIES

        self.p = p
        self.min = min
        self.max = max
        self.mode = mode

    def __call__(self, im):
        if isinstance(im, Image.Image) or isinstance(im, np.ndarray):
            im = F.to_tensor(im)

        if self.p < 1.0:
            mask = torch.bernoulli(torch.zeros(im.size(1), im.size(2)) + 1. - self.p).bool()
        else:
            mask = torch.zeros(im.size(1), im.size(2)).bool()

        if len(im.size())>2:
            mask = mask.unsqueeze(0).repeat(im.size(0),1,1)

        if self.mode == 'flip':
            im[mask] = self.max - im[mask]
        elif self.mode == 'drop':
            im[mask] = self.min

        return im
