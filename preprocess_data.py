import netCDF4
from datetime import date, timedelta
import xarray as xr
import dask.array as da
import pandas as pd
import numpy as np
import h5py
import torch
from numpy.lib.stride_tricks import sliding_window_view

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import h5py
import torchvision
from torchvision import transforms
from torch.utils.data import Subset

import os
import tempfile

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchgeo.datasets import EuroSAT
from torchvision.transforms import v2 as transforms


class Custom_Dataset(Dataset):
    def __init__(self, hdf5_data_file, im_name, label_name, exp_dataset='norm'):
        with h5py.File(hdf5_data_file,'r') as data:
            self.images = torch.Tensor(np.transpose(data[im_name],(0,3,1,2)))
            print('Input Shape: '+str(self.images.shape))
            if exp_dataset == 'mjo':
                self.labels = torch.Tensor(data[label_name][:])
            else:
                self.labels = torch.Tensor(data[label_name][:]) 

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx,:,:,:]
        label = self.labels[idx]
        sample = {'image': image, 'label': label}

        return image, label

class ChannelMNISTDataset(Dataset):
    def __init__(self, hdf5_data_file, im_name, label_name):
        with h5py.File(hdf5_data_file,'r') as data:
            self.images = torch.Tensor(data[im_name][:])
            self.labels = torch.Tensor(data[label_name][:])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        label = self.labels[idx]

        return image, label


def preprocess_ds(ds):
    ds = ds.sel(lat=slice(-15,15))
    ds = ds.sel(lon=slice(0,260))
    ds['time'] = ds.indexes['time'].normalize()
    return ds


def train_test_split(start_year=1981, end_year=2016, num_test_years=3, random_split=True):
    years = np.arange(start_year,end_year)
    if random_split==True:
        split_years = np.random.choice(years, size=num_test_years*2, replace=False)
        val_years = split_years[:num_test_years]
        test_years = split_years[num_test_years:]
        train_years = np.setdiff1d(years, test_years)
        train_years = np.setdiff1d(train_years, val_years)
    else:
        train_years = years[:-num_test_years*2]
        val_years = years[-num_test_years*2:-num_test_years]
        test_years = years[-num_test_years:]
    
    print('Train Years:')
    print(train_years)
    print('Val Years:')
    print(val_years)
    print('Test years: ')
    print(test_years)

    return train_years, val_years, test_years

def shuffle_data(x, y, time):
    index = np.arange(len(x))
    np.random.shuffle(index)
    #  return index for getting time 
    return x[index], y[index], time[index]


### function modified from B Toms 2020 downloaded from https://zenodo.org/records/3968896 
def process_mjo(data_dir, train_years, val_years, test_years, data_name, lead_time=1, shuffle=False, temporal=True, window_seq_len = 12, make_data=False, random_split=True, inactive_threshold=0.5):
    if temporal==False:
        window_seq_len=1

    #Load in the pre-processed MJO data that contains the dates, phases, and other information from the OMI MJO index
    MJO_OMI_data = np.loadtxt(data_dir+'MJO_OMI_data.txt', skiprows=0)

    #Only select the periods from 1980 through 2016, which is the period on which we train the neural networks
    MJO_OMI_data = MJO_OMI_data[ np.argwhere( (MJO_OMI_data[:,0] >= 1980) & (MJO_OMI_data[:,0] <= 2016) )[:,0] ]

    #And now extract the principal component and amplitude information to calculate the phase
    MJO_OMI_year = MJO_OMI_data[:,0].astype('int')
    MJO_OMI_pc1 = MJO_OMI_data[:,4]
    MJO_OMI_pc2 = MJO_OMI_data[:,5]
    MJO_OMI_amplitude = MJO_OMI_data[:,-1]

    #To be consistent with the broader MJO literature, we now adjust the OMI index to align with the RMM phase space
    MJO_OMI_pc1_toRMM = MJO_OMI_pc2
    MJO_OMI_pc2_toRMM = -1*MJO_OMI_pc1

    #We now need to calculate the phase of the MJO according to the RMM index
    MJO_OMI_phase = np.zeros_like(MJO_OMI_pc1_toRMM)

    for ind, i in enumerate(MJO_OMI_pc1_toRMM):
        if (MJO_OMI_pc1_toRMM[ind] < 0) & (MJO_OMI_pc2_toRMM [ind] < 0) & ( np.abs(MJO_OMI_pc1_toRMM[ind]) > np.abs(MJO_OMI_pc2_toRMM[ind]) ):
            MJO_OMI_phase[ind] = 1
        if (MJO_OMI_pc1_toRMM[ind] < 0) & (MJO_OMI_pc2_toRMM [ind] < 0) & ( np.abs(MJO_OMI_pc1_toRMM[ind]) < np.abs(MJO_OMI_pc2_toRMM[ind]) ):
            MJO_OMI_phase[ind] = 2
        if (MJO_OMI_pc1_toRMM[ind] > 0) & (MJO_OMI_pc2_toRMM [ind] < 0) & ( np.abs(MJO_OMI_pc1_toRMM[ind]) < np.abs(MJO_OMI_pc2_toRMM[ind]) ):
            MJO_OMI_phase[ind] = 3
        if (MJO_OMI_pc1_toRMM[ind] > 0) & (MJO_OMI_pc2_toRMM [ind] < 0) & ( np.abs(MJO_OMI_pc1_toRMM[ind]) > np.abs(MJO_OMI_pc2_toRMM[ind]) ):
            MJO_OMI_phase[ind] = 4
        if (MJO_OMI_pc1_toRMM[ind] > 0) & (MJO_OMI_pc2_toRMM [ind] > 0) & ( np.abs(MJO_OMI_pc1_toRMM[ind]) > np.abs(MJO_OMI_pc2_toRMM[ind]) ):
            MJO_OMI_phase[ind] = 5
        if (MJO_OMI_pc1_toRMM[ind] > 0) & (MJO_OMI_pc2_toRMM [ind] > 0) & ( np.abs(MJO_OMI_pc1_toRMM[ind]) < np.abs(MJO_OMI_pc2_toRMM[ind]) ):
            MJO_OMI_phase[ind] = 6
        if (MJO_OMI_pc1_toRMM[ind] < 0) & (MJO_OMI_pc2_toRMM [ind] > 0) & ( np.abs(MJO_OMI_pc1_toRMM[ind]) < np.abs(MJO_OMI_pc2_toRMM[ind]) ):
            MJO_OMI_phase[ind] = 7
        if (MJO_OMI_pc1_toRMM[ind] < 0) & (MJO_OMI_pc2_toRMM [ind] > 0) & ( np.abs(MJO_OMI_pc1_toRMM[ind]) > np.abs(MJO_OMI_pc2_toRMM[ind]) ):
            MJO_OMI_phase[ind] = 8

    MJO_OMI_phase[MJO_OMI_amplitude < inactive_threshold] = 0
    inactive_indices = (MJO_OMI_phase == 0)

    #MJO_OMI_phase = MJO_OMI_phase[~inactive_indices]
    #MJO_OMI_year = MJO_OMI_year[~inactive_indices]
        
    num_samples = len(MJO_OMI_phase)
    val_ind = np.where(np.isin(MJO_OMI_year, val_years))[0]
    test_ind = np.where(np.isin(MJO_OMI_year, test_years))[0]
    train_ind = np.where(np.isin(MJO_OMI_year, train_years))[0]

    y_train = np.asarray(MJO_OMI_phase[train_ind])
    y_val = np.asarray(MJO_OMI_phase[val_ind])
    y_test = np.asarray(MJO_OMI_phase[test_ind])
    ds = xr.open_mfdataset(data_dir+'*.nc', chunks='auto', preprocess=preprocess_ds)
    ds = ds.drop_vars('time_bnds')

    #ds = ds.drop_isel(time=np.asarray(inactive_indices))

    X_train = ds.sel(time=ds.time.dt.year.isin(train_years))
    X_test = ds.sel(time=ds.time.dt.year.isin(test_years))
    X_val = ds.sel(time=ds.time.dt.year.isin(val_years))

    olr_mean = X_train.olr.mean().values
    u200_mean = X_train.U.mean(dim=['time','lat','lon']).values[0]
    u850_mean = X_train.U.mean(dim=['time','lat','lon']).values[1]

    olr_std = X_train.olr.std().values
    u200_std = X_train.U.std(dim=['time','lat','lon']).values[0]
    u850_std = X_train.U.std(dim=['time','lat','lon']).values[1]

    print(olr_mean)
    print(olr_std)

    olr_min = X_train.olr.min().values
    u200_min = X_train.U.min(dim=['time','lat','lon']).values[0]
    u850_min = X_train.U.min(dim=['time','lat','lon']).values[1]

    olr_max = X_train.olr.max().values
    u200_max = X_train.U.max(dim=['time','lat','lon']).values[0]
    u850_max = X_train.U.max(dim=['time','lat','lon']).values[1]

    """
    X_train['olr'] = 2*(X_train.olr - olr_min)/(olr_max - olr_min) - 1
    X_val['olr'] = 2*(X_val.olr - olr_min)/(olr_max - olr_min) - 1
    X_test['olr'] = 2*(X_test.olr - olr_min)/(olr_max - olr_min) - 1

    X_train['U'][:,0,:,:] = 2*(X_train.U[:,0,:,:] - u200_min)/(u200_max - u200_min) - 1
    X_val['U'][:,0,:,:] = 2*(X_val.U[:,0,:,:] - u200_min)/(u200_max - u200_min) - 1
    X_test['U'][:,0,:,:] = 2*(X_test.U[:,0,:,:] - u200_min)/(u200_max - u200_min) - 1

    X_train['U'][:,1,:,:] = 2*(X_train.U[:,1,:,:] - u850_min)/(u850_max - u850_min) - 1
    X_val['U'][:,1,:,:] = 2*(X_val.U[:,1,:,:] - u850_min)/(u850_max - u850_min) - 1
    X_test['U'][:,1,:,:] = 2*(X_test.U[:,1,:,:] - u850_min)/(u850_max - u850_min) - 1
    """

    
    X_train['olr'] = (X_train.olr - olr_mean)/olr_std
    X_val['olr'] = (X_val.olr - olr_mean)/olr_std
    X_test['olr'] = (X_test.olr - olr_mean)/olr_std

    X_train['U'][:,0,:,:] = (X_train.U[:,0,:,:] - u200_mean)/u200_std
    X_val['U'][:,0,:,:] = (X_val.U[:,0,:,:] - u200_mean)/u200_std
    X_test['U'][:,0,:,:] = (X_test.U[:,0,:,:] - u200_mean)/u200_std

    X_train['U'][:,1,:,:] = (X_train.U[:,1,:,:] - u850_mean)/u850_std
    X_val['U'][:,1,:,:] = (X_val.U[:,1,:,:] - u850_mean)/u850_std
    X_test['U'][:,1,:,:] = (X_test.U[:,1,:,:] - u850_mean)/u850_std
    

    # Get the unique values and their counts
    unique_values, counts = np.unique(y_train, return_counts=True)
    # Print the results
    for value, count in zip(unique_values, counts):
        print(f"{value} occurs {count} times")

    unique_values, counts = np.unique(y_val, return_counts=True)
    # Print the results
    for value, count in zip(unique_values, counts):
        print(f"{value} occurs {count} times")

    unique_values, counts = np.unique(y_test, return_counts=True)
    # Print the results
    for value, count in zip(unique_values, counts):
        print(f"{value} occurs {count} times")

    #concatenate the channels to 2nd dimension
    train_data = np.concatenate((X_train.olr.values[:,:,:,np.newaxis], X_train.U.values[:,0,:,:,np.newaxis], X_train.U.values[:,1,:,:, np.newaxis]), axis=3)
    train_dates = np.array(X_train.time.values[:])
    train_time = np.array( [np.datetime_as_string(n,timezone='UTC').encode('utf-8') for n in train_dates] )
    val_data = np.concatenate((X_val.olr.values[:,:,:,np.newaxis], X_val.U.values[:,0,:,:,np.newaxis], X_val.U.values[:,1,:,:, np.newaxis]), axis=3)
    val_dates = np.array(X_val.time.values[:])
    val_time = np.array( [np.datetime_as_string(n,timezone='UTC').encode('utf-8') for n in val_dates] )
    test_data = np.concatenate((X_test.olr.values[:,:,:,np.newaxis], X_test.U.values[:,0,:,:,np.newaxis], X_test.U.values[:,1,:,:, np.newaxis]), axis=3)
    test_dates = np.array(X_test.time.values[:])
    test_time = np.array( [np.datetime_as_string(n,timezone='UTC').encode('utf-8') for n in test_dates] )

    print('concatented data ')

    if random_split ==True:
        shape = (window_seq_len)
        train_data_year = np.asarray(np.split(train_data,len(train_years)))
        train_y_year = np.asarray(np.split(y_train, len(train_years)))
        train_time_year = np.asarray(np.split(train_time, len(train_years)))
        v = sliding_window_view(train_data_year, shape, axis=1)
        v_time = sliding_window_view(train_time_year, shape, axis=1)
        if lead_time != 0:
            v = v[:,:-lead_time,:,:,:,:]
            v_time = v_time[:,:-lead_time,:]

        v_train = np.reshape(v, (v.shape[0]*v.shape[1], v.shape[2], v.shape[3], v.shape[4], v.shape[5]))
        v_train_time = np.reshape(v_time, (v_time.shape[0]*v_time.shape[1], v_time.shape[2]))
        num = window_seq_len+lead_time-1
        w = train_y_year[:,num:]
        w_train = np.reshape(w, (w.shape[0]*w.shape[1]))
        print(v_train.shape)
        print(v_train_time.shape)
        print(w_train.shape)

        val_data_year = np.asarray(np.split(val_data,len(val_years)))
        val_y_year = np.asarray(np.split(y_val, len(val_years)))
        val_time_year = np.asarray(np.split(val_time, len(val_years)))
        v = sliding_window_view(val_data_year, shape, axis=1)
        v_time = sliding_window_view(val_time_year, shape, axis=1)
        if lead_time != 0:
            v = v[:,:-lead_time,:,:,:,:]
            v_time = v_time[:,:-lead_time,:]
        v_val = np.reshape(v, (v.shape[0]*v.shape[1], v.shape[2], v.shape[3], v.shape[4], v.shape[5]))
        v_val_time = np.reshape(v_time, (v_time.shape[0]*v_time.shape[1], v_time.shape[2]))
        num = window_seq_len+lead_time-1
        w = val_y_year[:,num:]
        w_val = np.reshape(w, (w.shape[0]*w.shape[1]))
        print(v_val.shape)
        print(v_val_time.shape)
        print(w_val.shape)

        test_data_year = np.asarray(np.split(test_data,len(test_years)))
        test_y_year = np.asarray(np.split(y_test, len(test_years)))
        test_time_year = np.asarray(np.split(test_time, len(test_years)))
        v = sliding_window_view(test_data_year, shape, axis=1)
        v_time = sliding_window_view(test_time_year, shape, axis=1)
        if lead_time != 0:
            v = v[:,:-lead_time,:,:,:,:]
            v_time = v_time[:,:-lead_time,:]
        v_test = np.reshape(v, (v.shape[0]*v.shape[1], v.shape[2], v.shape[3], v.shape[4], v.shape[5]))
        v_test_time = np.reshape(v_time, (v_time.shape[0]*v_time.shape[1], v_time.shape[2]))
        num = window_seq_len+lead_time-1
        w = test_y_year[:,num:]
        w_test = np.reshape(w, (w.shape[0]*w.shape[1]))
        print(v_test.shape)
        print(v_test_time.shape)
        print(w_test.shape)

    if shuffle==True:
        v_train, w_train, v_train_time = shuffle_data(v_train, w_train, v_train_time)
        v_val, w_val, v_val_time = shuffle_data(v_val, w_val, v_val_time)
        v_test, w_test, v_test_time = shuffle_data(v_test, w_test, v_test_time)
        print('shuffled data')


    if make_data==True:
        
        v_train = np.squeeze(v_train)
        v_train_time = np.squeeze(v_train_time)
        v_val = np.squeeze(v_val)
        v_val_time = np.squeeze(v_val_time)
        v_test = np.squeeze(v_test)
        v_test_time = np.squeeze(v_test_time)
        with h5py.File(data_name,'a') as f:
            if 'train_images' in f:
                del f['train_images']
                del f['train_labels']
            if 'val_images' in f:
                del f['val_images']
                del f['val_labels']
                del f['test_images']
                del f['test_labels']
            if 'train_time' in f:
                del f['train_time']
                del f['val_time']
                del f['test_time']
            f.create_dataset("train_images", shape=v_train.shape, data=v_train)
            f.create_dataset("train_labels", shape=w_train.shape, data=w_train)
            f.create_dataset("train_time", shape=v_train_time.shape, data=v_train_time)

            f.create_dataset("val_images", shape=v_val.shape, data=v_val)
            f.create_dataset("val_labels", shape=w_val.shape, data=w_val)
            f.create_dataset("val_time", shape=v_val_time.shape, data=v_val_time)

            f.create_dataset("test_images", shape=v_test.shape, data=v_test)
            f.create_dataset("test_labels", shape=w_test.shape, data=w_test)
            f.create_dataset("test_time", shape=v_test_time.shape, data=v_test_time)

        print('created dataset')

def create_mjo_noise(noiseless_file_name, file_name):

    with h5py.File(noiseless_file_name,'r') as nf:
        train_data = nf['train_images'][:]
        val_data = nf['val_images'][:]
        test_data = nf['test_images'][:]
        train_time = nf['train_time'][:]

        train_labels = nf['train_labels'][:]
        val_labels = nf['val_labels'][:]
        test_labels = nf['test_labels'][:]

    
    train_rand = np.random.rand(train_data[:,:,:,0].shape[0],train_data[:,:,:,0].shape[1],train_data[:,:,:,0].shape[2],1)
    train_data = np.concatenate((train_data, train_rand), axis=3)
    print(train_data.shape)

    val_rand = np.random.rand(val_data[:,:,:,0].shape[0],val_data[:,:,:,0].shape[1],val_data[:,:,:,0].shape[2],1)
    val_data = np.concatenate((val_data, val_rand), axis=3)

    test_rand = np.random.rand(test_data[:,:,:,0].shape[0],test_data[:,:,:,0].shape[1],test_data[:,:,:,0].shape[2],1)
    test_data = np.concatenate((test_data, test_rand), axis=3)

    with h5py.File(file_name,'a') as f:

        del f['train_images']
        del f['val_images']
        del f['test_images']

        del f['train_labels']
        del f['val_labels']
        del f['test_labels']
        f.create_dataset("train_images", shape=train_data.shape, data=train_data)
        f.create_dataset("val_images", shape=val_data.shape, data=val_data)
        f.create_dataset("test_images", shape=test_data.shape, data=test_data)

        f.create_dataset("train_time", shape=train_time.shape, data=train_time)

        f.create_dataset("train_labels", shape=train_labels.shape, data=train_labels)
        f.create_dataset("val_labels", shape=val_labels.shape, data=val_labels)
        f.create_dataset("test_labels", shape=test_labels.shape, data=test_labels)

if __name__ == "__main__":
    
    #path to B Toms (2020) processed MJO data folder downloaded from https://zenodo.org/records/3968896 
    data_dir = 'data/raw_mjo_data/toms_etal_processed_data'

    random_split=True
    temporal=False
    window_seq_len = 1
    lead_time = 0
    data_name = "data/processed_mjo_data/processed_mjo.hdf5"
    

    train_years, val_years, test_years = train_test_split(1980, 2017, num_test_years=3, random_split=random_split)
    process_mjo(data_dir, train_years, val_years, test_years, data_name=data_name, lead_time=lead_time, shuffle=True, temporal=temporal, window_seq_len = window_seq_len, make_data=True, random_split=random_split, inactive_threshold=1)
    print(data_name)

    noiseless_file_name = "data/processed_mjo_data/processed_mjo.hdf5"
    file_name = "data/processed_mjo_data/processed_mjo_noise.hdf5"

    create_mjo_noise(noiseless_file_name, file_name)






    