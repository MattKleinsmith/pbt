import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import choose


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


def get_optimizer(model):
    """This is where users choose their optimizer and define the
       hyperparameter space they'd like to search."""
    optimizer_class = optim.SGD
    lr = choose(np.logspace(-5, 0, base=10))
    momentum = choose(np.linspace(0.1, .9999))
    return optimizer_class(model.parameters(), lr=lr, momentum=momentum)


DATA_DIR = "../../data/mnist/"
MODEL_CLASS = ConvNet
LOSS_FN = F.nll_loss
HYPERPARAM_NAMES = ["lr", "momentum"]  # This is unfortunate.
EPOCHS = 10
BATCH_SIZE = 64
POPULATION_SIZE = 15  # Number of models in a population
EXPLOIT_INTERVAL = 0.5  # When to exploit, in number of epochs
USE_SQLITE = True
