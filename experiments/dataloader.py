import os
import numpy as np
import torch

def load_user_data(dir: os.PathLike, val_user: int, device: str, users:int=8):
    """ Leave one user out cross validation """
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []

    for k in range(users):
        data = np.load(f"{dir}/data_{k}.npy")
        labels = np.load(f"{dir}/labels_{k}.npy")
        data = torch.tensor(data, dtype=torch.float32, device=device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        
        if k == val_user:
            val_data.append(data)
            val_labels.append(labels)
        else:
            train_data.append(data)
            train_labels.append(labels)
    
    train_data = torch.cat(train_data)
    train_labels = torch.cat(train_labels)
    val_data = torch.cat(val_data)
    val_labels = torch.cat(val_labels)

    print("Train data and labels shapes {}, {}".format(train_data.shape, train_labels.shape))
    print("Val data and labels shapes {}, {}".format(val_data.shape, val_labels.shape))

    data = {
        'train': (train_data, train_labels),
        'val': (val_data, val_labels),
        'test': (val_data, val_labels), # TODO
    }

    return data

def load_multisensor_data(dir: os.PathLike, val_user: int, body_parts: list, device: str, users:int=8):
    """
        TODO
        1. Implement multisensor data preprocessing for data and labels of each body part
    """
    train_data = {[] for bp in body_parts}
    train_labels = {[] for bp in body_parts}
    val_data = {[] for bp in body_parts}
    val_labels = {[] for bp in body_parts}

    for bp in body_parts:
        for k in range(users):
            data = np.load(f"{dir}/data_{k}_{bp}.npy") # TODO
            labels = np.load(f"{dir}/labels_{k}_{bp}.npy") # TODO
            data = torch.tensor(data, dtype=torch.float32, device=device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            
            if k == val_user:
                val_data[bp].append(data)
                val_labels[bp].append(labels)
            else:
                train_data[bp].append(data)
                train_labels[bp].append(labels)
    
    train_data = {torch.cat(train_data[bp]) for bp in body_parts}
    train_labels = {torch.cat(train_labels[bp]) for bp in body_parts}
    val_data = {torch.cat(val_data[bp]) for bp in body_parts}
    val_labels = {torch.cat(val_labels[bp]) for bp in body_parts}

    data = {
        'train': (train_data, train_labels),
        'val': (val_data, val_labels),
        'test': (val_data, val_labels), # TODO
    }

    return data


def load_merged_data(dir: os.PathLike, device: str):
    # load data
    train_data = np.load(f"{dir}/training_data.npy")
    train_labels = np.load(f"{dir}/training_labels.npy")
    train_data = torch.tensor(train_data, dtype=torch.float32, device=device)
    train_labels = torch.tensor(train_labels, dtype=torch.long, device=device)

    val_data = np.load(f"{dir}/val_data.npy")
    val_labels = np.load(f"{dir}/val_labels.npy")
    val_data = torch.tensor(val_data,  dtype=torch.float32, device=device)
    val_labels = torch.tensor(val_labels, dtype=torch.long, device=device)

    test_data = np.load(f"{dir}/testing_data.npy")
    test_labels = np.load(f"{dir}/testing_labels.npy")
    test_data = torch.tensor(test_data, dtype=torch.float32, device=device)
    test_labels = torch.tensor(test_labels, dtype=torch.long, device=device)

    data = {
        'train': (train_data, train_labels),
        'val': (val_data, val_labels),
        'test': (test_data, test_labels)
    }

    return data