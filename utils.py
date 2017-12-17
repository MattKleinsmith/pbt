# https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-utils-py

import random

import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    

def choose(x):
    return np.random.choice(x)


def plot_images(images, labels, preds=None):

    assert len(images) == len(labels) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, 0, :, :], interpolation='spline16', cmap='gray')
            
        label = str(labels[i])
        if preds is None:
            xlabel = label
        else:
            pred = str(preds[i])
            xlabel = "True: {0}\nPred: {1}".format(label, pred)
            
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()