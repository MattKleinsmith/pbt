from copy import deepcopy

import numpy as np
import torch
from torch.autograd import Variable
import tqdm

from utils import LossIsNaN, split_trn_val


class Trainer:

    def __init__(self, model, optimizer,
                 loss_fn=None, inputs=None, targets=None, batch_size=None,
                 valid_size=0.2, task_id=None):
        """Note: Trainer objects don't know about the database."""

        self.model = model
        self.optimizer = optimizer

        # Sometimes we only use a Trainer to load and save checkpoints.
        #   When that's the case, we don't need the following.
        if inputs:
            self.loss_fn = loss_fn
            self.inputs = inputs
            self.targets = targets
            self.batch_size = batch_size

            # Train-valid split
            num_examples = len(self.inputs)
            self.trn_indices, self.val_indices = \
                split_trn_val(num_examples, valid_size)

        self.task_id = task_id

    def save_checkpoint(self, checkpoint_path):
        checkpoint = dict(model_state_dict=self.model.state_dict(),
                          optim_state_dict=self.optimizer.state_dict())
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])

    def train(self, second_half, seed_for_shuffling):
        np.random.seed(seed_for_shuffling)
        np.random.shuffle(self.trn_indices)  # BUG: This doesn't do anything.
        print("Error: Your data isn't being shuffled. Fix train function or use your own.")
        num_batches = int(np.ceil(len(self.trn_indices) / self.batch_size))
        batch_indices = tqdm.tqdm(range(num_batches),
                                  desc='Train (task %d)' % self.task_id,
                                  ncols=80, leave=True)
        for k in batch_indices:
            if second_half and k < num_batches//2:
                continue
            inp = self.inputs[k*self.batch_size:(k+1)*self.batch_size]
            target = self.targets[k*self.batch_size:(k+1)*self.batch_size]
            self.step(inp, target)

    def step(self, inp, target):
        """Forward pass and backpropagation"""
        self.model.train()
        inp = Variable(torch.from_numpy(inp).cuda())
        target = Variable(torch.from_numpy(target).long().cuda())
        output = self.model(inp)
        loss = self.loss_fn(output, target)
        if np.isnan(float(loss.data[0])):
            print("Loss is NaN.")
            raise LossIsNaN
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def eval(self, interval_id):
        """Evaluate model on the provided validation or test set."""
        self.model.eval()
        correct = 0
        num_batches = int(np.ceil(len(self.val_indices) / (self.batch_size)))
        batch_indices = tqdm.tqdm(range(num_batches),
                                  desc='Eval (interval %d)' % interval_id,
                                  ncols=80, leave=True)
        for k in batch_indices:
            inp = self.inputs[k*self.batch_size:(k+1)*self.batch_size]
            target = self.targets[k*self.batch_size:(k+1)*self.batch_size]
            inp = Variable(torch.from_numpy(inp).cuda(), volatile=True)
            target = Variable(torch.from_numpy(target).long().cuda())
            output = self.model(inp)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        accuracy = 100. * correct / len(self.val_indices)
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
