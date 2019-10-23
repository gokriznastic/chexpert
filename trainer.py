import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os

class Trainer():
    def __init__(self, model, train_loader, val_loader):
        """ Initialze the model and dataloaders for the trainer

        Arguments:
            model : the model to be trained
            train_loader : DataLoader containing training data
            val_loader : DataLoader containing validation data
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.losses = {'train':[], 'validation':[]}

    def get_device(self):
        """ Function to check for GPU if available, if not return CPU as device

        Returns:
            device : gpu or cpu detected by torch
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        return self.device

    def compile(self, lr, loss_fn, scheduler=False):
        """ Function to initialize loss, optimizer and scheduler for the model to be trained

        Arguments:
            lr : learning rate of the model
            loss : the loss function to be minimized, could be from torch.nn or could be user-defined

        Keyword Arguments:
            scheduler : whether to use scheduler if loss plateaus (default: False)
        """
        self.learning_rate = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        self.loss_fn = loss_fn

        if (scheduler):
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, verbose=True, min_lr=1e-6)
            # self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)

    def load_checkpoint(self, log_path, model_path):
        """ Function to load checkpoints from a previously saved model
        Arguments:
            log_path : path to saved log file
            model_path : path to saved model

        Returns:
            start_epoch : the epoch from which to resume the training
            model : the model with loaded weights
            optimizer : the optimizer with the previous state
            scheduler : the scheduler with the previous state
            losses : the dictionary populated with loss values from previous epochs
        """
        # Note: Input model & optimizer should be pre-defined. This routine only updates their states
        start_epoch = 0

        self.log_path = log_path
        self.model_path = model_path

        print(self.log("\nLoading checkpoint from '{}'\n".format(model_path)))

        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        try:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            pass

        self.losses = checkpoint['loss']
        val_min = min(self.losses['validation'])

        print(self.log("--> Loaded checkpoint from '{}'\nResuming training from epoch {}\n\n"
                    .format(model_path, start_epoch)))

        return start_epoch, self.model, self.optimizer, self.scheduler, self.losses, val_min
        # return start_epoch, self.model, self.optimizer, self.losses, val_min


    def epoch_train(self, print_every=50):
        model = self.model
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        device = self.device

        train_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):

            data, target = data.float().to(device), target.float().to(device)

            optimizer.zero_grad()

            pred = model(data)
            loss = loss_fn(pred, target)

            train_loss += ((1 / (batch_idx + 1)) * (loss.item()/data.size(0) - train_loss))
            loss.backward()
            optimizer.step()

            if (batch_idx % print_every == 0):
                print('Epoch {}\tBatch [{}/{}]\t\tTraining Loss: {}'.format(self.epoch+1, batch_idx+1, len(self.train_loader), train_loss))

        return train_loss

    def epoch_val(self):
        model = self.model
        loss_fn = self.loss_fn
        device = self.device

        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.float().to(device), target.float().to(device)

                val_pred = model(data)
                val_loss = loss_fn(val_pred, target)

                valid_loss += ((1 / (batch_idx + 1)) * (val_loss.item()/data.size(0) - valid_loss))

        return valid_loss

    def train(self, n_epochs, batch_size, log_path=None, model_path=None):
        """ Function to the train the model

        Arguments:
            n_epochs : number of epochs for which the model should be trained
            batch_size : batch_size of the DataLoaders
            log_path : path to saved log file
            model_path : path to saved model

        Returns:
            model : trained model
            losses : the dictionary containing the loss values
        """
        start = datetime.now()

        model = self.model
        optimizer = self.optimizer
        try:
            scheduler = self.scheduler
        except:
            pass
        losses = self.losses
        valid_loss_min = np.Inf
        start_epoch = 0

        if (log_path is None and model_path is None):
            self.log_path = str(start.strftime('%d-%m-%Y-%H:%M:%S')+'_train_log')
            self.model_path = '{}_model.pt'.format(start.strftime('%d-%m-%Y-%H:%M:%S'))
            self.log('Learning rate: {}, Batch size: {}\n\n'.format(self.learning_rate, batch_size))

        elif (os.path.exists(model_path) and os.path.exists(log_path)):
            # start_epoch, model, optimizer, scheduler, losses = self.load_checkpoint(log_path, model_path)
            start_epoch, model, optimizer, scheduler, losses, valid_loss_min = self.load_checkpoint(log_path, model_path)
            valid_loss_min = min(losses['validation'])

        else:
            print('[!] Specified model path or log path does not exist.')
            return

        # loss_fn = self.loss_fn

        checkpoint = {}

        for self.epoch in range(start_epoch, start_epoch+n_epochs):

            # self.epoch = epoch

            train_loss = self.epoch_train(print_every=500)
            valid_loss = self.epoch_val()

            print(self.log('\nEpoch: [{}/{}] \tTraining Loss: {:.5f} \tValidation Loss: {:.5f}\n'.format(
                                                    self.epoch+1, start_epoch+n_epochs, train_loss, valid_loss)))
            print('-'*100)

            #####----CHECKPOINTING----#####
            if (valid_loss < valid_loss_min):
                print(self.log("Saving model.  Validation loss:... {:.5f} --> {:.5f}\n".format(valid_loss_min, valid_loss)))
                print('*'*100)
                valid_loss_min = valid_loss

                checkpoint['model_state_dict'] = model.state_dict()
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                try:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                except:
                    pass
                print()

            try:
                scheduler.step(valid_loss)
            except:
                pass

            losses['train'].append(train_loss)
            losses['validation'].append(valid_loss)

            checkpoint['epoch'] = self.epoch
            checkpoint['loss'] = losses
            torch.save(checkpoint, self.model_path)

            self.draw_loss_curve('{}_losses.png'.format(self.model_path.split('_')[0]))

        end = datetime.now()
        time = str(end - start).split('.')[0]
        print(self.log("\nCompleted training in {}\n".format(time)))

        return model, losses

    def log(self, info):
        """ Function to create and update the log file

        Arguments:
            info : the update information to write on the log file

        Returns:
            info : the logged information to be printed while training
        """
        log_path = self.log_path

        if not os.path.exists(log_path):
            file = open(log_path, 'w')
            file.write(info)
            file.close()
        else:
            file = open(log_path, 'a')
            file.write(info)
            file.close()

        return info.strip('\n')

    def draw_loss_curve(self, fpath, losses=None):
        """ Function to generate loss curve for the training process

        Arguments:
            fpath : the filepath to save the loss curve in
        """
        if losses is None:
            losses = self.losses
        # plt.ylim([0,2])
        plt.plot(losses['train'], label='Training loss')
        plt.plot(losses['validation'], label='Validation loss')
        plt.legend()
        plt.savefig(fpath)
        # plt.show()
        plt.close()