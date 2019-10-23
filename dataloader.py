import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms

import torch
from torch.utils.data import Dataset
from utils import multihot
from image_utils import clahe, get_aug


# def getImagesLabels(filename):
#     df = pd.read_csv(filename)

#     X = df['Path']
#     y = df[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']]

#     return np.asarray(X), np.asarray(y)

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
        # image = Image.open(image_name).convert('RGB')
        image = clahe(image_name)

        label = self.labels[index]
        gt = self.gt[index]

        # index = random.choice([0,1,2,3])
        # if (self.test==False):
        #     image = get_aug(image, index)

        # plt.imshow(image)
        # plt.show()

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
