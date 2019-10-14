import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from torchvision import transforms

import torch
from torch.utils.data import Dataset
from utils import multihot

random_state = np.random.RandomState(0)

def getImagesLabels(filename, policy):
    """
    filename: path to the csv file containing all the imagepaths and associated labels
    """
    df = pd.read_csv(filename)
    relevant_cols = ['Path', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    df = df[relevant_cols]
    df = df.replace(np.nan, 0.0)

    if policy == 'zeros':
        df = df.replace(-1.0, 0.0)
    elif policy == 'ones':
        df = df.replace(-1.0, 1.0)
    elif policy == 'both':
        df['Cardiomegaly'] = df['Cardiomegaly'].replace(-1.0, 0.0)
        df['Consolidation'] = df['Consolidation'].replace(-1.0, 0.0)

        df['Atelectasis'] = df['Atelectasis'].replace(-1.0, 1.0)
        df['Edema'] = df['Edema'].replace(-1.0, 1.0)
        df['Pleural Effusion'] = df['Pleural Effusion'].replace(-1.0, 1.0)

    elif policy == 'multi' or policy == 'ignore':
        df = df.replace(-1.0, 2.0)

    # df = df[df['Path'].str.contains('frontal')] ###

    X = df['Path']
    y = df[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']]

    return np.asarray(X), np.asarray(y)

def elastic_transform(image, alpha, sigma):
    image = np.asarray(image)

    if len(image.shape) < 3:
        image = image.reshape(image.shape[0], image.shape[1], -1)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z, (-1, 1))

    distorted_image = map_coordinates(image, indices, order=0, mode='reflect')

    return Image.fromarray(distorted_image.reshape(image.shape))

def get_aug(image, index):
    if (index == 0):
        return image
    else:
        alpha = np.random.uniform(10,100)
        sigma = np.random.uniform(10,15)
        input_transformed= elastic_transform(image, alpha=alpha, sigma=sigma)
        return  input_transformed


class CheXpertDataset(Dataset):
    def __init__(self, image_list, labels, transform=None, test=False):
        """
        image_list: list of paths containing images
        labels: corresponding multi-class labels of image_list
        transform: optional transform to be applied on a sample.
        """
        self.image_names = image_list
        self.gt = labels
        if (len(np.unique(labels)) > 2):
            self.labels = multihot(labels)
        else:
            self.labels = labels

        self.transform = transform
        self.test = test

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""

        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        gt = self.gt[index]

        # index = random.randint(0,3)
        # if (self.test==False):
        #     image = get_aug(image, index)

        if self.transform is not None:
            image = self.transform(image)
        else:
            self.transform = transform = transforms.Compose([
                    transforms.Resize((320, 320)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
            image = self.transform(image)
        if self.test:
            return image, torch.FloatTensor(gt)
        else:
            return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


# x, y = getImagesLabels('CheXpert-v1.0-small/train.csv', 'both')
# dataset = CheXpertDataset(x,y)
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)

# it = iter(train_loader)
# t = next(it)
# print(t[0].shape, t[1].shape)
