import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class discreteDataset(Dataset):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __len__(self):
        return np.prod(self.shape)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        indices = np.unravel_index(idx, self.shape)

        return indices  # , self.dat[indices]


class discreteDatasetMasked(Dataset):
    def __init__(self, dat, mask):
        super().__init__()
        self.dat = dat
        self.mask = mask  # mask equals 1 OUTSIDE of valid data
        self.Nmask = np.sum(~mask)
        self.Nall = np.prod(dat.shape)
        self.int2ext = np.arange(self.Nall)[~self.mask.flatten()]

    def __len__(self):
        return self.Nmask

    def __getitem__(self, idx_int):
        #         if torch.is_tensor(idx_int):
        #             idx_int = idx_int.tolist()

        #         print('idx_int ', idx_int)
        idx_ext = self.int2ext[idx_int]
        indices_ext = np.unravel_index(idx_ext, self.dat.shape)

        return indices_ext  # , self.dat[indices_ext]


def tensor_friendly_sampler(dataset, batch_size=512):
    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(dataset),
        batch_size=batch_size,
        drop_last=False,
    )  # VERY essential! without custom sampler somewhy it will work 100x times slower
    return sampler


def tensor_friendly_dataloader(dataset, batch_size=512):
    sampler = tensor_friendly_sampler(dataset, batch_size=batch_size)
    dl = DataLoader(dataset, sampler=sampler)
    return dl


# class DicreteDataset(Dataset):
#     def __init__(self, shape):
#         self.shape = shape

#     def __len__(self):
#         return np.prod(self.shape)

#     def __getitem__(self, idx):
#         indices = np.unravel_index(idx, self.shape)
#         return indices


# class PartialDataset(Dataset):
#     def __init__(self, shape, mask):
#         super().__init__()
#         self.shape = shape
#         self.mask = mask
#         self.indices = np.arange(np.prod(shape))[mask.flatten()]

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         idx = self.indices[idx]
#         indices = np.unravel_index(idx, self.shape)
#         return indices