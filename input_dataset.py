import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from os.path import isfile, isdir
from scipy.io import loadmat
from tqdm import tqdm
import numpy as np


def download_data():

    data_dir = 'data/'

    if not isdir(data_dir):
        raise Exception("Data directory doesn't exist!")

    class DLProgress(tqdm):
        last_block = 0

        def hook(self, block_num=1, block_size=1, total_size=None):
            self.total = total_size
            self.update((block_num - self.last_block) * block_size)
            self.last_block = block_num

    if not isfile(data_dir + "train_32x32.mat"):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='SVHN Training Set') as pbar:
            urlretrieve(
                'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
                data_dir + 'train_32x32.mat',
                pbar.hook)

    if not isfile(data_dir + "test_32x32.mat"):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='SVHN Training Set') as pbar:
            urlretrieve(
                'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
                data_dir + 'test_32x32.mat',
                pbar.hook)

    trainset = loadmat(data_dir + 'train_32x32.mat')
    testset = loadmat(data_dir + 'test_32x32.mat')
    return trainset, testset


def scale(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    x = ((x - x.min()) / (255 - x.min()))

    # scale to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x

def view_samples(epoch, samples, nrows, ncols, figsize=(5, 5)):
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,
                             sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        ax.axis('off')
        img = ((img - img.min()) * 255 / (img.max() - img.min())).astype(np.uint8)
        ax.set_adjustable('box-forced')
        im = ax.imshow(img)

    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axes








