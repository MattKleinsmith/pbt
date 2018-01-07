import argparse
import mkl
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from psycopg2 import ProgrammingError

from data_loaders import get_train_valid_loader
from trainer import Trainer
from utils import (create_table, update_task, get_max_of_db_column,
                   insert_into_table, get_a_task, ExploitationNeeded,
                   LossIsNaN, get_task_ids_and_scores, PopulationFinished,
                   get_interval_ids)


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


if __name__ == "__main__":
    # TODO: Does this help?
    nproc = mkl.get_max_threads()  # e.g. 12
    mkl.set_num_threads(nproc)

    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="Selects GPU with the given ID. IDs are those shown in nvidia-smi.")  # noqa
    parser.add_argument("-r", "--resume", type=int, default=None, help="Resume work on the most recently created population. The most recently created population will be used unless the population ID is given.")  # noqa
    parser.add_argument("-e", "--exploiter", action="store_true", help="Set this process as the exploiter. It will be responsible for running the exploit step over the entire population at the end of each interval.")  # noqa
    args = parser.parse_args()
    gpu = args.gpu
    resume = args.resume
    exploiter = args.exploiter

    DATA_DIR = "../../data/mnist/"
    MODEL_CLASS = ConvNet
    OPTIMIZER_CLASS = optim.SGD
    HYPERPARAM_NAMES = ["lr", "momentum"]
    # HP ranges and scales hardcoded into trainer.py

    EPOCHS = 10
    BATCH_SIZE = 64

    POPULATION_SIZE = 15  # Number of models in a population  # Maybe 20
    EXPLOIT_INTERVAL = 0.5  # When to exploit, in number of epochs

    interval_limit = int(np.ceil(EPOCHS / EXPLOIT_INTERVAL))

    # Postgres
    DB_ENV_VAR_NAMES = ['PGDATABASE', 'PGUSER', 'PGPORT', 'PGHOST']
    db_parameters = [os.environ[var_name] for var_name in DB_ENV_VAR_NAMES]
    db_connect_str = "dbname={} user={} port={} host={}".format(*db_parameters)

    checkpoint_str = "checkpoints/pop-%03d_task-%03d.pth"

    # Load data
    # MNIST is hard-coded into the DataLoaders, and CUDA is assumed.
    loaders = get_train_valid_loader(DATA_DIR, batch_size=BATCH_SIZE,
                                     random_seed=123)
    train_loader, valid_loader = loaders

    if resume is None:
        # Fill population table with a new population
        table_name = "populations"
        try:
            latest_population_id = get_max_of_db_column(db_connect_str,
                                                        table_name,
                                                        "population_id")
            population_id = latest_population_id + 1
        except ProgrammingError:
            # Create population table
            command = """
                      CREATE TABLE populations (
                            population_id INTEGER,
                            task_id INTEGER,
                            interval_id INTEGER,
                            ready_for_exploitation BOOLEAN,
                            active BOOLEAN,
                            score REAL,
                            seed_for_shuffling INTEGER
                      )
                      """
            create_table(db_connect_str, command)
            population_id = 0
        for task_id in range(POPULATION_SIZE):
            key_value_pairs = dict(population_id=population_id,
                                   task_id=task_id,
                                   interval_id=0,
                                   ready_for_exploitation=False,
                                   active=False,
                                   score=None,
                                   seed_for_shuffling=123)
            insert_into_table(db_connect_str, table_name, key_value_pairs)
        print("\nPopulation added to populations table. Population ID: %s" %
              population_id)
    else:
        population_id = resume
    # Train each available task for an interval
    while True:
        # Find a task that's incomplete and inactive, and set it to active
        try:
            task = get_a_task(db_connect_str, population_id, interval_limit)
            task_id, interval_id, seed_for_shuffling = task
        except PopulationFinished:
            break
        except ExploitationNeeded:
            if exploiter:
                interval_ids = get_interval_ids(db_connect_str, population_id)
                # Sorted by scores, desc
                task_ids, scores = get_task_ids_and_scores(db_connect_str,
                                                           population_id)
                print("Exploiting. Interval ID: %s Best score: %.2f" %
                      (max(interval_ids), max(scores)))
                seed_for_shuffling = np.random.randint(10**5)
                fraction = 0.20
                cutoff = int(np.ceil(fraction * len(task_ids)))
                top_ids = task_ids[:cutoff]
                bottom_ids = task_ids[cutoff:]
                for bottom_id in bottom_ids:
                    top_id = np.random.choice(top_ids)
                    top_trainer = Trainer(model_class=MODEL_CLASS)
                    top_checkpoint_path = (checkpoint_str %
                                           (population_id, top_id))
                    top_trainer.load_checkpoint(top_checkpoint_path)
                    bot_trainer = Trainer(model_class=MODEL_CLASS)
                    bot_checkpoint_path = (checkpoint_str %
                                           (population_id, bottom_id))
                    bot_trainer.exploit_and_explore(top_trainer,
                                                    HYPERPARAM_NAMES)
                    key_value_pairs = dict(
                        ready_for_exploitation=False,
                        score=None,
                        seed_for_shuffling=seed_for_shuffling)
                    update_task(db_connect_str, population_id, bottom_id,
                                key_value_pairs)
                for top_id in top_ids:
                    key_value_pairs = dict(
                        ready_for_exploitation=False,
                        seed_for_shuffling=seed_for_shuffling)
                    update_task(db_connect_str, population_id, top_id,
                                key_value_pairs)
                continue
            else:
                print("Waiting for exploiter to finish.")
                time.sleep(1)
                continue

        # Train
        with torch.cuda.device(gpu):
            trainer = Trainer(model_class=MODEL_CLASS,
                              train_loader=train_loader,
                              valid_loader=valid_loader)
            checkpoint_path = (checkpoint_str %
                               (population_id, task_id))
            if os.path.isfile(checkpoint_path):
                trainer.load_checkpoint(checkpoint_path)
            interval_is_odd = interval_id % 2 == 1
            score = None
            try:
                try:
                    trainer.train(second_half=interval_is_odd)
                except LossIsNaN:
                    print("Setting score to -1.")
                    score = -1
                if score != -1:
                    score = trainer.eval()
                    trainer.save_checkpoint(checkpoint_path)
                key_value_pairs = dict(interval_id=interval_id+1,
                                       ready_for_exploitation=True,
                                       active=False,
                                       score=score)
                update_task(db_connect_str, population_id, task_id,
                            key_value_pairs)
            except KeyboardInterrupt:
                # Don't save work.
                key_value_pairs = dict(active=False)
                update_task(db_connect_str, population_id, task_id,
                            key_value_pairs)
                break
