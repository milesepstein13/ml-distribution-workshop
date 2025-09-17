from torch.utils.data import Dataset
import numpy as np
import torch


class LazyWeatherDataset(Dataset):
    def __init__(self, xr_dataset, y):
        self.ds = xr_dataset.load()  # full xarray dataset
        self.y = torch.tensor(y.values, dtype=torch.float32)            # (332, target_dim) torch tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        day_data = self.ds.sel(day=self.ds.day[idx])

        
        tod = day_data.sizes["tod"]
        channels_per_tod = []

        for var in day_data.data_vars:
            da = day_data[var]
            if "level" in da.dims:
                da = da.transpose("level", "latitude", "longitude", "tod")
                data = da.values  # shape: (L, H, W, T)
                data = data.reshape(da.sizes["level"], da.sizes["latitude"], da.sizes["longitude"], tod)
            else:
                da = da.transpose("latitude", "longitude", "tod")
                data = da.values[np.newaxis, ...]  # add a channel axis: (1, H, W, T)
            channels_per_tod.append(data)

        x = np.concatenate(channels_per_tod, axis=0)  # (C, H, W, T)
        x = torch.tensor(x, dtype=torch.float32)


        y = self.y[idx].unsqueeze(0)
        return x, y