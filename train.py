import sys

import numpy as np
from torchvision import transforms

import torch
import torch.nn as nn
from dataloader import CheXpertDataset, getImagesLabels
from network import DenseNet121
from torch.utils.data import DataLoader
from trainer import Trainer

# batchSize = int(sys.argv[1])
# lr = float(sys.argv[2])
# epochs = int(sys.argv[3])
# num_classes = int(sys.argv[4])
# policy = sys.argv[5]
policy = 'both'

train_csv = 'CheXpert-v1.0-small/train.csv'
# train_csv = 'train.csv'

cropSize = 320

transform = transforms.Compose([
                    transforms.RandomResizedCrop(cropSize),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])

train_images, train_labels = getImagesLabels(train_csv, policy)
# train_images, train_labels = getImagesLabels(train_csv)

val_images, val_labels = train_images[180001:], train_labels[180001:]
train_images, train_labels = train_images[:180000], train_labels[:180000]

# val_images, val_labels = train_images[270000:], train_labels[270000:]
# train_images, train_labels = train_images[:270000], train_labels[:270000]

# val_images, val_labels = train_images[8000:10000], train_labels[8000:10000]
# train_images, train_labels = train_images[:8000], train_labels[:8000]

print('No. of images:\n\t|--Training: {}\n\t|--Validation: {}\n'.format(len(train_images), len(val_images)))

train_dataset = CheXpertDataset(train_images, train_labels, transform)
val_dataset = CheXpertDataset(val_images, val_labels, transform)

### hyperparameters ###
batchSize = 32
lr = 0.0001
epochs = 30
num_classes = 5

### data loaders ###
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True,  num_workers=24, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False,  num_workers=24, pin_memory=True)

### initializing the model ###
model = DenseNet121(num_classes)
model = torch.nn.DataParallel(model)
model.cuda()

### training ###
trainer = Trainer(model, train_loader, val_loader)
device = trainer.get_device()

# moving the model to device
model.to(device)

# defining the losses with class weights

# weights = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
# class_weights = torch.FloatTensor(weights).cuda()
# pos_weights = [0.45, 5.43, 1, 0.19, 7.17, 1, 0.24, 13.72, 1, 0.47, 2.85, 1, 0.83, 1.48, 1]
# pos_weights = torch.FloatTensor(pos_weights).cuda()
# criterion = nn.BCEWithLogitsLoss(weight=class_weights, pos_weight=pos_weights)

if(policy == 'zeros'):
    pos_weights = [5.43, 7.17, 13.71, 2.85, 1.48] # zeros
elif(policy == 'ones'):
    pos_weights = [2.2, 5.35, 4.11, 2.11, 1.21] # ones
elif(policy == 'both'):
    pos_weights = [2.2, 7.17, 13.71, 2.11, 1.21] # ones
    # pos_weights = [3.265, 10.269, 16.251, 3.968, 2.019] #chexpert+NIH

pos_weights = torch.FloatTensor(pos_weights).cuda()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

trainer.compile(lr, criterion, scheduler=True)

trainer.train(epochs, batchSize)