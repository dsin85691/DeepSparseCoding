import torch
from torchvision import datasets, transforms

import DeepSparseCoding.utils.data_processing as dp


def load_dataset(params):
    if(params.dataset.lower() == 'mnist'):
        preprocessing_pipeline = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)) # channels last
            ]
        if params.standardize_data:
            preprocessing_pipeline.append(
                transforms.Lambda(lambda x: dp.standardize(x, eps=params.eps)[0]))
        # Load dataset
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
    else:
        assert False, (f'Supported datasets are ["mnist"], not {dataset_name}')
    params.epoch_size = len(train_loader.dataset)
    if(not hasattr(params, 'num_val_images')):
        params.num_val_images = len(test_loader.dataset)
    if(not hasattr(params, 'num_test_images')):
        params.num_test_images = len(test_loader.dataset)
    params.data_shape = list(next(iter(train_loader))[0].shape)[1:]
    return (train_loader, val_loader, test_loader, params)
