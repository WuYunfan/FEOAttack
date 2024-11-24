import numpy as np
import torch
from dataset import get_dataset


def dropit_dataset(dataset, ratio):
    for user in range(dataset.n_users):
        num_items = int(len(dataset.train_data[user]) * ratio)
        train_data = np.random.choice(list(dataset.train_data[user]), num_items, replace=False)
        dataset.train_data[user] = set(train_data)
        num_items = int(len(dataset.val_data[user]) * ratio)
        val_data = np.random.choice(list(dataset.val_data[user]), num_items, replace=False)
        dataset.val_data[user] = set(val_data)


def main():
    device = torch.device('cpu')
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Gowalla/time', 'device': device}
    dataset = get_dataset(dataset_config)
    dropit_dataset(dataset, 0.5)
    dataset.output_dataset('data/Gowalla/partial')


if __name__ == '__main__':
    main()