import os
import numpy as np
import torch

def load_data(dir, device):
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
    test_data = torch.tensor(test_data[:500], dtype=torch.float32, device=device)
    test_labels = torch.tensor(test_labels[:500], dtype=torch.long, device=device)

    data = {
        'train': (train_data, train_labels),
        'val': (val_data, val_labels),
        'test': (test_data, test_labels)
    }

    return data

