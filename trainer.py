from copy import deepcopy

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import tqdm

from utils import choose


class Trainer:

    def __init__(self, model, test_loader, valid_loader, id):

        self.model = model
        self.test_loader = test_loader
        self.valid_loader = valid_loader
        self.id = id

        self.lr = choose(np.logspace(-5, 0, base=10))
        self.momentum = choose(np.linspace(0.1, .9999))
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.lr, momentum=self.momentum)
        self.parent = self.id
        self.history = {"lr": [], "momentum": [],
                        "test_accuracy": [], "valid_accuracy": []}

    def save_history(self, save_hyperparams):
        if save_hyperparams:
            self.history['lr'].append(self.lr)
            self.history['momentum'].append(self.momentum)
        # self.history['test_accuracy'].append(self.eval(self.test_loader))
        self.history['valid_accuracy'].append(self.eval(self.valid_loader))

    def step(self, data, target):
        """Forward pass and backpropagation"""
        self.model.train()
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = self.model(data)
        loss = F.nll_loss(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def eval(self, data_loader):
        """Evaluate model on the provided validation or test set."""
        self.model.eval()
        correct = 0
        data_iter = iter(data_loader)
        batches = tqdm.trange(len(data_iter),
                              ncols=80, desc="Batches (eval)")
        for _ in batches:
            try:
                data, target = next(data_iter)
            except StopIteration:
                break
            data, target = (Variable(data.cuda(), volatile=True),
                            Variable(target.cuda()))
            output = self.model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # len(data_loader.dataset) is incorrect if you use a sampler.
        #   e.g. SubsetRandomSampler in data_loaders.py
        num_examples = len(data_loader) * data_loader.batch_size
        accuracy = 100. * correct / num_examples
        return accuracy

    def exploit_and_explore(self, best_trainer, hyperparam_names,
                            perturb_factors=[1.2, 0.8]):
        """Copy parameters from the best model and the hyperparameters
           and running averages from the corresponding optimizer."""
        # Copy model parameters
        best_model = best_trainer.model
        best_state_dict = deepcopy(best_model.state_dict())
        self.model.load_state_dict(best_state_dict)
        # Copy optimizer state (including hyperparameters)
        best_optimizer = best_trainer.optimizer
        best_state_dict = deepcopy(best_optimizer.state_dict())
        self.optimizer.load_state_dict(best_state_dict)
        # Assumption: Same LR for each param group
        self.lr = self.optimizer.param_groups[0]['lr']
        self.momentum = self.optimizer.param_groups[0]['momentum']
        # Perturb hyperparameters
        for hyperparam_name in hyperparam_names:
            perturb = np.random.choice(perturb_factors)
            if hyperparam_name == 'lr':
                self.lr *= perturb
                hp = self.lr
            elif hyperparam_name == 'momentum':
                self.momentum *= perturb
                hp = self.momentum
            for param_group in self.optimizer.param_groups:
                param_group[hyperparam_name] = hp
