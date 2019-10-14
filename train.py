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

cropSize = 320

transform = transforms.Compose([
                    transforms.RandomResizedCrop(cropSize),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])

train_images, train_labels = getImagesLabels(train_csv, policy)

val_images, val_labels = train_images[180001:], train_labels[180001:]
train_images, train_labels = train_images[:180000], train_labels[:180000]

# val_images, val_labels = train_images[8000:10000], train_labels[8000:10000]
# train_images, train_labels = train_images[:8000], train_labels[:8000]

print('No. of images:\n\t|--Training: {}\n\t|--Validation: {}\n'.format(len(train_images), len(val_images)))

train_dataset = CheXpertDataset(train_images, train_labels, transform)
val_dataset = CheXpertDataset(val_images, val_labels, transform)

### hyperparameters ###
batchSize = 32
lr = 0.00001
epochs = 20
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
pos_weights = torch.FloatTensor(pos_weights).cuda()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

trainer.compile(lr, criterion, scheduler=True)

trainer.train(epochs, batchSize)

# def log(info, log_path):
#     if not os.path.exists(log_path):
#         file = open(log_path, 'w')
#         file.write(info)
#         file.close()
#     else:
#         file = open(log_path, 'a')
#         file.write(info)
#         file.close()

#     return info.strip('\n')

# def draw_loss_curve(fpath, losses=None):
#     # plt.ylim([0,2])
#     plt.plot(losses['train'], label='Training')
#     plt.plot(losses['validation'], label='Validation')
#     plt.legend()
#     plt.savefig(fpath)
#     # plt.show()
#     plt.close()

# from torch import optim
# from datetime import datetime
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
# criterion = nn.BCELoss(size_average=True)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def train(n_epochs, model, batch_size, criterion, optimizer, log_path=None, model_path=None):
#     losses = {'train':[], 'validation':[]}
#     start = datetime.now()

#     model = model
#     optimizer = optimizer
#     try:
#         scheduler = scheduler
#     except:
#         pass
#     losses = losses

#     if (log_path is None and model_path is None):
#         log_path = str(start.strftime('%d-%m-%Y-%H:%M:%S')+'_train_log')
#         model_path = '{}_model.pt'.format(start.strftime('%d-%m-%Y-%H:%M:%S'))
#         log('Learning rate: {}, Batch size: {}\n\n'.format(lr, batch_size), log_path)
#     else:
#         print('[!] Specified model path or log path does not exist.')

#     # loss_fn = loss_fn
#     valid_loss_min = np.Inf
#     start_epoch = 0

#     checkpoint = {}
#     train_loss = 0.0
#     valid_loss = 0.0

#     for epoch in range(start_epoch, start_epoch+n_epochs):
#         model.train()
#         for batch_idx, (data, target) in enumerate(train_loader):

#             data, target = data.float().to(device), target.float().to(device)
#             # data, target = data.float().to(device), target.long().to(device)

#             optimizer.zero_grad()

#             pred = model(data)
#             loss = criterion(pred, target)

#             train_loss += ((1 / (batch_idx + 1)) * (loss.item()/data.size(0) - train_loss))
#             loss.backward()
#             optimizer.step()

#             if (batch_idx % 50 == 0):
#                 print('Epoch {}\tBatch [{}/{}]\t\tTraining Loss: {}'.format(epoch+1, batch_idx+1, len(train_loader), train_loss))

#         model.eval()
#         with torch.no_grad():
#             for batch_idx, (data, target) in enumerate(val_loader):
#                 data, target = data.float().to(device), target.float().to(device)
#                 # data, target = data.float().to(device), target.long().to(device)

#                 val_pred = model(data)
#                 val_loss = criterion(val_pred, target)

#                 valid_loss += ((1 / (batch_idx + 1)) * (val_loss.item()/data.size(0) - valid_loss))

#         print(log('Epoch: [{}/{}] \tTraining Loss: {:.5f} \tValidation Loss: {:.5f}\n'.format(
#                                                 epoch+1, start_epoch+n_epochs, train_loss, valid_loss), log_path))
#         print('-'*100)

#         #####----CHECKPOINTING----#####
#         if (valid_loss < valid_loss_min):
#             print(log("Saving model.  Validation loss:... {:.5f} --> {:.5f}\n".format(valid_loss_min, valid_loss), log_path))
#             print('*'*100)
#             valid_loss_min = valid_loss

#             checkpoint['model_state_dict'] = model.state_dict()
#             checkpoint['optimizer_state_dict'] = optimizer.state_dict()
#             try:
#                 checkpoint['scheduler_state_dict'] = scheduler.state_dict()
#             except:
#                 pass
#             print()

#         try:
#             scheduler.step(valid_loss)
#         except:
#             pass

#         losses['train'].append(train_loss)
#         losses['validation'].append(valid_loss)

#         checkpoint['epoch'] = epoch
#         checkpoint['loss'] = losses
#         torch.save(checkpoint, model_path)

#         draw_loss_curve('{}_losses.png'.format(model_path.split('_')[0]), losses )

#     end = datetime.now()
#     time = str(end - start).split('.')[0]
#     print(log("\nCompleted training in {}\n".format(time), log_path))

#     return model, losses

# train(15, model, batchSize, criterion, optimizer)
