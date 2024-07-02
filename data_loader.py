import csv
import os

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split


class Data(Dataset):
    def __init__(self, data_dir='.\\dataset\\MSTAR', transform=transforms.ToTensor()):
        self.data_dir = data_dir
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        for csv_file in ['train.csv', 'test.csv']:
            csv_path = os.path.join(self.data_dir, csv_file)
            with open(csv_path, 'r') as f:
                context = csv.reader(f, delimiter=',')
                for path, label in context:
                    self.img_paths.append(path)
                    self.img_labels.append(np.array(int(label)))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_label = self.img_labels[idx]
        img = Image.open(img_path).convert('RGB')
        transforms_img = self.transform(img)
        # print(img_label)
        # transforms_label = self.transform(img_label)
        return transforms_img, img_label


class Dataloader(object):
    def __init__(self, data_dir='./dataset/MSTAR/', batch_size=16, num_workers=1, transform=transforms.ToTensor(), split=0.8):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.split = split
        self.dataset = Data(self.data_dir, transform=self.transform)

    def get_loader(self):
        train_size = int(self.split*len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_dataloader, test_dataloader

