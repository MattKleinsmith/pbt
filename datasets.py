import os.path as osp

import numpy as np
import bcolz
from torchvision import datasets, transforms


def bcolz_save(path, np_array):
    c = bcolz.carray(np_array, rootdir=path, mode='w')
    c.flush()
    print("Saved to " + path)


get_mnist = False
if get_mnist:
    DATA_DIR = "../../data/mnist/"
    TRN_INPUTS_BCOLZ_PATH = osp.join(DATA_DIR, "trn_inputs.bcolz")
    TRN_TARGETS_BCOLZ_PATH = osp.join(DATA_DIR, "trn_targets.bcolz")
    TST_INPUTS_BCOLZ_PATH = osp.join(DATA_DIR, "tst_inputs.bcolz")
    TST_TARGETS_BCOLZ_PATH = osp.join(DATA_DIR, "tst_targets.bcolz")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),
                                                         (0.3081,))])
    trn_dataset_tensor = datasets.MNIST(DATA_DIR, train=True, download=True,
                                        transform=transform)
    tst_dataset_tensor = datasets.MNIST(DATA_DIR, train=False, download=True,
                                        transform=transform)

    trn_inputs_np = np.array([x.numpy() for x, y in list(trn_dataset_tensor)])
    trn_targets_np = np.array([y for x, y in trn_dataset_tensor])
    tst_inputs_np = np.array([x.numpy() for x, y in list(tst_dataset_tensor)])
    tst_targets_np = np.array([y for x, y in tst_dataset_tensor])

    bcolz_save(TRN_INPUTS_BCOLZ_PATH, trn_inputs_np)
    bcolz_save(TRN_TARGETS_BCOLZ_PATH, trn_targets_np)
    bcolz_save(TST_INPUTS_BCOLZ_PATH, tst_inputs_np)
    bcolz_save(TST_TARGETS_BCOLZ_PATH, tst_targets_np)

get_cifar10 = False
if get_cifar10:
    DATA_DIR = "../../data/cifar10/"
    TRN_INPUTS_BCOLZ_PATH = osp.join(DATA_DIR, "trn_inputs.bcolz")
    TRN_TARGETS_BCOLZ_PATH = osp.join(DATA_DIR, "trn_targets.bcolz")
    TST_INPUTS_BCOLZ_PATH = osp.join(DATA_DIR, "tst_inputs.bcolz")
    TST_TARGETS_BCOLZ_PATH = osp.join(DATA_DIR, "tst_targets.bcolz")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))])
    trn_dataset_tensor = datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                          transform=transform)
    tst_dataset_tensor = datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                          transform=transform)

    trn_inputs_np = np.array([x.numpy() for x, y in list(trn_dataset_tensor)])
    trn_targets_np = np.array([y for x, y in trn_dataset_tensor])
    tst_inputs_np = np.array([x.numpy() for x, y in list(tst_dataset_tensor)])
    tst_targets_np = np.array([y for x, y in tst_dataset_tensor])

    bcolz_save(TRN_INPUTS_BCOLZ_PATH, trn_inputs_np)
    bcolz_save(TRN_TARGETS_BCOLZ_PATH, trn_targets_np)
    bcolz_save(TST_INPUTS_BCOLZ_PATH, tst_inputs_np)
    bcolz_save(TST_TARGETS_BCOLZ_PATH, tst_targets_np)
