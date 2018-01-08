import os.path as osp

import numpy as np
import bcolz
from torchvision import datasets, transforms


def bcolz_save(path, np_array):
    c = bcolz.carray(np_array, rootdir=path, mode='w')
    c.flush()
    print("Saved to " + path)


DATA_DIR = "../../data/mnist/"
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

trn_dataset_tensor = datasets.MNIST(DATA_DIR, train=True, download=True,
                                    transform=transform)
trn_dataset_tensor = list(trn_dataset_tensor)
# TODO: See whether bcolz can save PyTorch Tensors well
trn_inputs_np = np.array([x.numpy() for x, y in trn_dataset_tensor])
path = osp.join(DATA_DIR, "trn_inputs.bcolz")
bcolz_save(path, trn_inputs_np)
trn_targets_np = np.array([y for x, y in trn_dataset_tensor])
path = osp.join(DATA_DIR, "trn_targets.bcolz")
bcolz_save(path, trn_targets_np)

tst_dataset_tensor = datasets.MNIST(DATA_DIR, train=False, download=True,
                                    transform=transform)
tst_dataset_tensor = list(tst_dataset_tensor)
tst_inputs_np = np.array([x.numpy() for x, y in tst_dataset_tensor])
path = osp.join(DATA_DIR, "tst_inputs.bcolz")
bcolz_save(path, tst_inputs_np)
tst_targets_np = np.array([y for x, y in tst_dataset_tensor])
path = osp.join(DATA_DIR, "tst_targets.bcolz")
bcolz_save(path, tst_targets_np)
