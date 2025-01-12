import os
import sys

import numpy as np
import torch
from torchvision import datasets, transforms

ROOT_DIR = os.path.dirname(os.getcwd())
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import DeepSparseCoding.utils.data_processing as dp
import DeepSparseCoding.datasets.synthetic as synthetic


class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def load_dataset(params):
    new_params = {}
    if(params.dataset.lower() == 'mnist'):
        preprocessing_pipeline = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)) # channels last
            ]
        if params.standardize_data:
            preprocessing_pipeline.append(
                transforms.Lambda(lambda x: dp.standardize(x, eps=params.eps)[0]))
        if params.rescale_data_to_one:
            preprocessing_pipeline.append(
                transforms.Lambda(lambda x: dp.rescale_data_to_one(x, eps=params.eps, samplewise=True)[0]))
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=params.data_dir, train=True, download=True,
            transform=transforms.Compose(preprocessing_pipeline)),
            batch_size=params.batch_size, shuffle=params.shuffle_data,
            num_workers=0, pin_memory=False)
        val_loader = None
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=params.data_dir, train=False, download=True,
            transform=transforms.Compose(preprocessing_pipeline)),
            batch_size=params.batch_size, shuffle=params.shuffle_data,
            num_workers=0, pin_memory=False)
    elif(params.dataset.lower() == 'dsprites'):
        root = os.path.join(*[params.data_dir])
        dsprites_file = os.path.join(*[root, 'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'])
        if not os.path.exists(dsprites_file):
            import subprocess
            print(f'Now downloading the dsprites-dataset to {root}/dsprites')
            subprocess.call(['./scripts/download_dsprites.sh', f'{root}'])
            print('Finished')
        data = np.load(dsprites_file, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset
        train_data = dset(**train_kwargs)
        train_loader = torch.utils.data.DataLoader(train_data,
            batch_size=params.batch_size,
            shuffle=params.shuffle_data,
            num_workers=0,
            pin_memory=False)
        val_loader = None
        test_loader = None
    elif (params.dataset.lower() == 'cifar10'):
        preprocessing_pipeline = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.permute(1, 2, 0))  # channels last
        ]
        if params.standardize_data:
            preprocessing_pipeline.append(
                transforms.Lambda(lambda x: dp.standardize(x, eps=params.eps)[0]))
        if params.rescale_data_to_one:
            preprocessing_pipeline.append(
                transforms.Lambda(lambda x: dp.rescale_data_to_one(x, eps=params.eps, samplewise=True)[0]))
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=params.data_dir, train=True, download=True,
                           transform=transforms.Compose(preprocessing_pipeline)),
            batch_size=params.batch_size, shuffle=params.shuffle_data,
            num_workers=0, pin_memory=False)
        val_loader = None
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=params.data_dir, train=False, download=True,
                           transform=transforms.Compose(preprocessing_pipeline)),
            batch_size=params.batch_size, shuffle=params.shuffle_data,
            num_workers=0, pin_memory=False)
    elif(params.dataset.lower() == 'synthetic'):
        preprocessing_pipeline = [transforms.ToTensor(),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)) # channels last
            ]
        train_loader = torch.utils.data.DataLoader(
            synthetic.SyntheticImages(params.epoch_size, params.data_edge_size, params.dist_type,
            params.rand_state, params.num_classes,
            transform=transforms.Compose(preprocessing_pipeline)),
            batch_size=params.batch_size, shuffle=params.shuffle_data,
            num_workers=0, pin_memory=False)
        val_loader = None
        test_loader = None
        new_params["num_pixels"] = params.data_edge_size**2
    else:
        assert False, (f'Supported datasets are ["mnist", "dsprites", "synthetic"], not {dataset_name}')
    new_params = {}
    new_params['epoch_size'] = len(train_loader.dataset)
    if(not hasattr(params, 'num_val_images')):
        if val_loader is None:
            new_params['num_val_images'] = 0
        else:
            new_params['num_val_images'] = len(val_loader.dataset)
    if(not hasattr(params, 'num_test_images')):
        if test_loader is None:
            new_params['num_test_images'] = 0
        else:
            new_params['num_test_images'] = len(test_loader.dataset)
    new_params['data_shape'] = list(next(iter(train_loader))[0].shape)[1:]
    return (train_loader, val_loader, test_loader, new_params)
