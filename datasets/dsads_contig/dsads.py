import torch
from torch.utils.data import Dataset
import os
import numpy as np
from utils.setup_funcs import PROJECT_ROOT


class DSADS(Dataset):
    """ PyTorch dataset class for the preprocessed DSADS dataset

    Parameters
    ----------

    root_dir: str
        global path to the dsads preprocessed data

    users: list
        list of users to load data for, subset of [1,2,3,4,5,6,7,8]

    body_parts: list
        list of body parts to get sensor channels from, subset of ['torso','right_arm','left_arm','right_leg','left_leg']

    train: bool
        whether to get the training data

    val: bool
        whether to get the validation data

    """
    def __init__(self, root_dir,train=False,val=False,test=False):
        
        # accelerometer channels
        self.body_part_map = {'right_leg':[0,1,2]}
        
        self.train = train
        self.val = val
        if train:
            prefix = f"{root_dir}/training"
        elif val:
            prefix = f"{root_dir}/val"
        elif test:
            prefix = f"{root_dir}/testing"

        self.body_parts = np.array(self.body_part_map['right_leg'])

        self.raw_data = np.load(f"{prefix}_data.npy")
        self.raw_labels = np.load(f"{prefix}_labels.npy")
        self.window_idxs = np.load(f"{prefix}_window_idxs.npy")
        self.window_labels = np.load(f"{prefix}_window_labels.npy")


    def preprocess(self,rescale=None,normalize=False):
        """ rescaling and normalization using training statistics

        Parameters
        ----------

        rescale: list 
            [min,max] range to rescale

        normalize: bool
            whether to subtract mean and divide by standard deviation

        """

        # use training min,max on test data
        if rescale is not None:
            all_data = self.raw_data
            if self.train == True:
                self.min_val = np.min(all_data)
                self.max_val = np.max(all_data)
                self.raw_data = ((self.raw_data-self.min_val)/(self.max_val-self.min_val))*(rescale[1]-rescale[0]) + rescale[0]
        
        # use train mean, std on test data
        if normalize == True:
            all_data = self.raw_data
            if self.train == True:
                self.mean = np.mean(all_data,axis=0)
                self.std = np.std(all_data,axis=0)
                # self.raw_data = (self.raw_data-self.mean)/(self.std + 1e-5)
            self.raw_data = (self.raw_data-self.mean)/(self.std + 1e-5)

    def __getitem__(self, idx):

        # get the window idxs
        start = self.window_idxs[idx]
        # print(f"idx: {idx}, count: {count}, user_i: {user_i}, start,end: {start},{end}")
        
        # get the data window
        X = self.raw_data[int(start):int(start)+8,:]

        # get the label
        Y = self.window_labels[idx]

        # return the sample and the class
        # transpose because we want (C x L), i.e. each row is a new channel and columns are time
        return torch.tensor(X.T).float(), torch.tensor(Y).long()

    def __len__(self):
        return len(self.window_labels)
    

def load_dsads_person_dataset(batch_size):
    root_dir = os.path.join(PROJECT_ROOT,"../data/dsads/merged_preprocess")

    train_ds = DSADS(root_dir,train=True)
    # print(train_ds.raw_data.mean(axis=0))
    train_ds.preprocess(None,True)
    # print(train_ds.raw_data.mean(axis=0))

    val_ds = DSADS(root_dir,val=True)
    # val_ds.min_val = train_ds.min_val
    # val_ds.max_val = train_ds.max_val
    val_ds.mean = train_ds.mean
    val_ds.std = train_ds.std
    # print(val_ds.raw_data.mean(axis=0))
    val_ds.preprocess(None,True)
    # print(val_ds.raw_data.mean(axis=0))

    test_ds = DSADS(root_dir,test=True)
    # test_ds.min_val = train_ds.min_val
    # test_ds.max_val = train_ds.max_val
    test_ds.mean = train_ds.mean
    test_ds.std = train_ds.std
    test_ds.preprocess(None,True)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False,drop_last=True,num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=False,drop_last=True,num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=False,drop_last=True,num_workers=4)

    return train_loader, val_loader, test_loader