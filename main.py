from copy import deepcopy
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import tqdm

from data_loaders import get_train_valid_loader, get_test_loader
from population import Population


plt.switch_backend('agg')
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (10, 5)
# sns.palplot(sns.color_palette())  # To preview palette


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def plot_accuracies(pbt_experimenter, random_experimenter, accuracy_key, seed):
    _, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2)
    pbt_experimenter.plot_accuracies(ax1, 'pbt', "purple", accuracy_key)
    random_experimenter.plot_accuracies(ax2, 'random', "purple",
                                        accuracy_key)
    plt.tight_layout(pad=1)
    plt.savefig("outputs/acc_%s.png" % seed)


def plot_hyperparams(pbt_experimenter, random_experimenter, accuracy_key,
                     seed):
    # TODO: DRY
    _, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2)
    pbt_experimenter.plot_hyperparams(ax1, 'pbt', "purple", accuracy_key)
    random_experimenter.plot_hyperparams(ax2, 'random', "purple",
                                         accuracy_key)
    plt.tight_layout(pad=1)
    plt.savefig("outputs/hp_%s.png" % seed)


if __name__ == "__main__":
    DATA_DIR = "../../data/mnist/"
    HOURS_AVAILABLE = 1
    EXPERIMENTS_PER_HOUR = 2  # About 3 on a 1080 Ti, with these settings
    EPOCHS = 1
    BATCH_SIZE = 64
    POPULATION_SIZE = 2  # Number of models in a population
    EXPLOIT_INTERVAL = 0.5  # When to exploit, in number of epochs

    number_of_experiments = int(EXPERIMENTS_PER_HOUR * HOURS_AVAILABLE)
    model_class = ConvNet
    optimizer_class = optim.SGD
    hyperparam_names = ["lr", "momentum"]
    # MNIST is hard-coded into the DataLoaders, and CUDA is assumed.
    train_loader, valid_loader = get_train_valid_loader(DATA_DIR,
                                                        batch_size=BATCH_SIZE,
                                                        random_seed=123,
                                                        show_sample=False)
    test_loader = get_test_loader(DATA_DIR, batch_size=BATCH_SIZE*2)
    population_kwargs = dict(model_class=model_class,
                             optimizer_class=optimizer_class,
                             hyperparam_names=hyperparam_names,
                             train_loader=train_loader,
                             valid_loader=valid_loader,
                             test_loader=test_loader,
                             population_size=POPULATION_SIZE,
                             epochs=EPOCHS,
                             exploit_interval=EXPLOIT_INTERVAL)
    sns.set_palette(sns.diverging_palette(10, 129, s=99, l=50, sep=100,  # noqa
                    center="dark", n=POPULATION_SIZE))

    seeds = tqdm.trange(number_of_experiments, ncols=80, desc="Seeds")
    for seed in seeds:
        pbt_population = Population(**population_kwargs,  # noqa
                                    seed=seed, do_pbt=True)
        pbt_population.train("pbt")
