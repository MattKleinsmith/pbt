from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import tqdm

from utils import choose, LossIsNaN


class Trainer:

    def __init__(self, model_class, train_loader=None, valid_loader=None):
        """Note: Trainer objects don't know about the database."""

        self.model = model_class().cuda()
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.loss_fn = F.nll_loss
        lr = choose(np.logspace(-5, 0, base=10))
        momentum = choose(np.linspace(0.1, .9999))
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=lr, momentum=momentum)

    def save_checkpoint(self, checkpoint_path):
        checkpoint = dict(model_state_dict=self.model.state_dict(),
                          optim_state_dict=self.optimizer.state_dict())
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])

    def train(self, second_half):
        train_iter = iter(self.train_loader)
        num_batches = len(train_iter) // 2
        if second_half:
            for _ in range(num_batches):
                next(train_iter)
        batch_indices = tqdm.trange(num_batches, ncols=80, desc="Batches")
        for _ in batch_indices:
            try:
                data, target = next(train_iter)
            except StopIteration:
                break
            self.step(data, target)

    def step(self, data, target):
        """Forward pass and backpropagation"""
        self.model.train()
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = self.model(data)
        loss = self.loss_fn(output, target)
        if np.isnan(float(loss.data[0])):
            print("Loss is NaN.")
            raise LossIsNaN
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def eval(self, data_loader=None):
        """Evaluate model on the provided validation or test set."""
        self.model.eval()
        correct = 0
        if data_loader is None:
            data_loader = self.valid_loader
        data_iter = iter(data_loader)
        batch_indices = tqdm.trange(len(data_iter),
                                    ncols=80, desc="Batches (eval)")
        with torch.no_grad():
            for _ in batch_indices:
                try:
                    data, target = next(data_iter)
                except StopIteration:
                    break
                data, target = (Variable(data.cuda()), Variable(target.cuda()))
                output = self.model(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # len(data_loader.dataset) is incorrect if you use a sampler.
        #   e.g. SubsetRandomSampler in data_loaders.py
        num_examples = len(data_loader) * data_loader.batch_size
        accuracy = 100. * correct / num_examples
        return accuracy

    def exploit_and_explore(self, better_trainer, hyperparam_names,
                            perturb_factors=[1.2, 0.8]):
        """Copy parameters from the better model and the hyperparameters
           and running averages from the corresponding optimizer."""
        # Copy model parameters
        better_model = better_trainer.model
        better_state_dict = deepcopy(better_model.state_dict())
        self.model.load_state_dict(better_state_dict)
        # Copy optimizer state (includes hyperparameters and running averages)
        better_optimizer = better_trainer.optimizer
        better_state_dict = deepcopy(better_optimizer.state_dict())
        self.optimizer.load_state_dict(better_state_dict)
        # Assumption: Same LR and momentum for each param group
        # Perturb hyperparameters
        for hyperparam_name in hyperparam_names:
            perturb = np.random.choice(perturb_factors)
            for param_group in self.optimizer.param_groups:
                param_group[hyperparam_name] *= perturb
