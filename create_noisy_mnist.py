import numpy as np
from urllib import request
import gzip
import pickle
import scipy
import matplotlib.pyplot as plt

filename = [
["training_images","train-images-idx3-ubyte"],
["test_images","t10k-images-idx3-ubyte"],
["training_labels","train-labels-idx1-ubyte"],
["test_labels","t10k-labels-idx1-ubyte"]
]

root_dir = '/data/MNIST/raw'

def add_noise(X1, X2, debug=False):
    out1 = []
    out2 = []

    N = X1.shape[0]
    angles = np.random.uniform(low=-45.0, high=45.0, size=[N])
    bg = np.random.uniform(size=X2.shape) * 0.4

    for i in range(N):
        x1 = X1[i].reshape(28, 28)
        ang = angles[i]
        # print(ang)
        img1 = scipy.ndimage.rotate(x1, ang)
        out1.append(img1.reshape(-1))

        x2 = X2[i]
        img2 = x2 + bg[i]
        img2[img2 > 1.0] = 1.0
        out2.append(img2)

        if debug:
            fig, axs = plt.subplots(4)
            axs[0].imshow(x1, cmap='gray')
            axs[1].imshow(img1, cmap='gray')
            axs[2].imshow(x2.reshape(28, 28), cmap='gray')
            axs[3].imshow(img2.reshape(28, 28), cmap='gray')
            plt.show()
    return np.concatenate(out1, 0), np.concatenate(out2, 0)


def create_mnist():
    mnist = {}
    for name in filename[:2]:
        with open(root_dir+'/'+name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with open(root_dir+'/'+name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    noisy_mnist = {}

    # Training and validation.
    X, T = mnist["training_images"], mnist["training_labels"]
    train_x1 = []
    train_x2 = []
    train_labels = []
    valid_x1 = []
    valid_x2 = []
    valid_labels = []
    for i in range(10):
        indices1 = np.where(T==i)[0]
        indices2 = np.array(indices1)
        labs = np.ones_like(indices1) * i
        np.random.shuffle(indices2)

        train_x1.append(X[indices1[:-500]])
        train_x2.append(X[indices2[:-500]])
        train_labels.append(labs[:-500])

        valid_x1.append(X[indices1[-500:]])
        valid_x2.append(X[indices2[-500:]])
        valid_labels.append(labs[-500:])

    train_x1 = np.concatenate(train_x1, 0) / 255.0
    train_x2 = np.concatenate(train_x2, 0) / 255.0
    train_labels = np.concatenate(train_labels, 0)
    N_train = train_labels.shape[0]

    valid_x1 = np.concatenate(valid_x1, 0) / 255.0
    valid_x2 = np.concatenate(valid_x2, 0) / 255.0
    valid_labels = np.concatenate(valid_labels, 0)
    N_valid = valid_labels.shape[0]

    # Test.
    X, T = mnist["test_images"], mnist["test_labels"]
    test_x1 = []
    test_x2 = []
    test_labels = []
    for i in range(10):
        indices1 = np.where(T==i)[0]
        indices2 = np.array(indices1)
        np.random.shuffle(indices2)
        labs = np.ones_like(indices1) * i

        test_x1.append(X[indices1])
        test_x2.append(X[indices2])
        test_labels.append(labs)

    test_x1 = np.concatenate(test_x1, 0) / 255.0
    test_x2 = np.concatenate(test_x2, 0) / 255.0
    test_labels = np.concatenate(test_labels, 0)
    N_test = test_labels.shape[0]

    # Create noisy views.
    train_x1, train_x2 = add_noise(train_x1, train_x2)
    valid_x1, valid_x2 = add_noise(valid_x1, valid_x2)
    test_x1, test_x2 = add_noise(test_x1, test_x2)

    with open("noisy_mnist_two_views.pkl", 'wb') as f:
        noisy_mnist = {'train_x1': train_x1, 'train_x2': train_x2, 'N_train': N_train,
            'valid_x1': valid_x1, 'valid_x2': valid_x2, 'N_valid': N_valid,
            'test_x1': test_x1, 'test_x2': test_x2, 'N_test': N_test,
        }
        pickle.dump(mnist,f)
    print("Save complete.")


if __name__ == '__main__':
    create_mnist()