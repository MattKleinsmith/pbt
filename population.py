import numpy as np
import tqdm

from trainer import Trainer
from utils import set_seed


class Population:

    def __init__(self, model_class, optimizer_class,
                 train_loader, valid_loader, test_loader,
                 hyperparam_names, population_size, epochs, exploit_interval,
                 seed, do_pbt):

        set_seed(seed)

        self.model_class = model_class
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.hyperparam_names = hyperparam_names
        self.population_size = population_size
        self.epochs = epochs
        self.do_pbt = do_pbt

        self.trainers = self.make_trainers()
        self.exploitations_per_epoch = exploit_interval
        self.exploit_interval = self.make_exploit_interval(exploit_interval)
        self.half_interval = self.exploit_interval//2

    def make_trainers(self):
        trainers = []
        for i in range(self.population_size):
            trainer = Trainer(
                model=self.model_class().cuda(),
                test_loader=self.test_loader,
                valid_loader=self.valid_loader,
                id=i)
            trainers.append(trainer)
        return trainers

    def make_exploit_interval(self, interval):
        # Convert interval from epochs to batches
        interval = len(self.train_loader) * interval
        # Make it an even number
        interval = (interval-1) if interval % 2 else interval
        return interval

    def save_history(self, save_hyperparams):
        for trainer in self.trainers:
            trainer.save_history(save_hyperparams=save_hyperparams)

    def sort_trainers_by_accuracy(self, accuracy_key):
        """Ascending order.
           Only use after training, e.g. for plotting. Otherwise you'll be
           modifying the order of a list that's being looped through
           in self.run."""
        self.trainers = sorted(self.trainers,
                               key=lambda t: t.history[accuracy_key][-1])

    def get_best_trainer(self, final_selection):
        """
        When searching for hyperparameters (e.g. via random search)
        make sure final_selection==False. Use final_selection==True when
        you're finished searching for hyperparameters, and are ready to use
        the test set to find the best model. You can only use the test set
        for selection once. Once you do, you've consumed it and it is no longer
        a proper test set. However, a consumed test set can still be used as
        training data or validation data.
        """
        accuracy_key = 'test_accuracy' if final_selection else 'valid_accuracy'
        best_accuracy = 0
        best_trainer = None
        for trainer in self.trainers:
            accuracy = trainer.history[accuracy_key][-1]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_trainer = trainer
        return best_trainer

    def train(self, runname):
        best_trainer_ids = []
        self.save_history(save_hyperparams=True)
        epochs = tqdm.trange(self.epochs, ncols=80,
                             desc="Epochs (%s)" % runname)
        for epoch_i in epochs:
            train_iter = iter(self.train_loader)
            batches = tqdm.trange(len(train_iter),
                                  ncols=80, desc="Batches")
            for batch_i in batches:
                try:
                    data, target = next(train_iter)
                except StopIteration:
                    break
                at_half_interval = (batch_i+1) % self.half_interval == 0
                at_exploit_interval = (batch_i+1) % self.exploit_interval == 0
                if at_half_interval:
                    self.save_history(save_hyperparams=at_exploit_interval)
                if at_exploit_interval:
                    best_trainer = self.get_best_trainer(final_selection=False)
                    best_trainer_ids.append(best_trainer.id)
                for trainer in self.trainers:
                    condition = at_exploit_interval and self.do_pbt
                    if condition and trainer != best_trainer:
                        trainer.exploit_and_explore(best_trainer,
                                                    self.hyperparam_names)
                    trainer.step(data, target)

    def plot_accuracies(self, ax, run_name, best_color, accuracy_key):
        self.sort_trainers_by_accuracy(accuracy_key)
        epoch_values = [i * self.exploitations_per_epoch/2 for i in
                        range(len(self.trainers[0].history[accuracy_key]))]
        accuracies = []
        for trainer in self.trainers[:-1]:
            accuracies_w_i = trainer.history[accuracy_key]
            ax.plot(epoch_values, accuracies_w_i)
            accuracies += accuracies_w_i

        # Give the best result a special color
        accuracies_w_i = self.trainers[-1].history[accuracy_key]
        ax.plot(epoch_values, accuracies_w_i, color=best_color)
        accuracies += accuracies_w_i

        ax.set_xlabel('epochs')
        ax.set_ylabel(accuracy_key)
        ax.set_xlim(-0.125, max(epoch_values))
        ax.set_ylim(0, 100)
        ax.set_title(run_name + " (best acc: %0.3f)" % max(accuracies))

    def plot_hyperparams(self, ax, run_name, best_color, accuracy_key):
        self.sort_trainers_by_accuracy(accuracy_key)
        for trainer in self.trainers[:-1]:
            history = trainer.history
            n = len(history['lr'])
            dot_sizes = np.array(list((range(1, n+1)))) * 10
            ax.scatter(np.log10(history['lr']), history['momentum'],
                       s=dot_sizes, edgecolors='black', linewidth=0.15)

        # Give the best result a special color and shape
        history = self.trainers[-1].history
        n = len(history['lr'])
        scalar = 10
        dot_sizes = np.append([scalar * 2],
                              np.array(list((range(2, n)))) * scalar)
        dot_sizes = np.append(dot_sizes, [n * scalar * 2])
        ax.scatter(np.log10(history['lr']), history['momentum'], s=dot_sizes,
                   edgecolors='black', linewidth=0.15, color=best_color,
                   marker='*')

        ax.set_title(run_name)
        ax.set_xlabel('learning rate')
        ax.set_ylabel('momentum')
        ax.set_xlim(-6, 0)
        ax.set_ylim(0, 1)
