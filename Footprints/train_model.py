import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import models
import datasets


class Trainer(nn.Module):
    def __init__(self, model, training_path, validation_path, batch_size=8, train_eval_fraction=0.2,
                 learning_rate=0.001, use_GPU=False):
        super(Trainer, self).__init__()
        self.model = model
        self.training_data = datasets.FootprintsDataset(training_path)
        self.validation_data = datasets.FootprintsDataset(validation_path)
        self.use_GPU = use_GPU

        self.train_dataloader = DataLoader(self.training_data, batch_size=batch_size, shuffle=True)
        # train eval dataloader used to load large portion of training data for loss calc
        self.train_eval_dataloader = DataLoader(self.training_data,
                                                batch_size=int(train_eval_fraction*len(self.training_data)),
                                                shuffle=True)
        self.val_dataloader = DataLoader(self.validation_data, batch_size=len(self.validation_data))
        self.training_loss = []
        self.validation_loss = []

        # Move model to GPU
        if self.use_GPU:
            self.gpu = torch.device('cuda')
            print("Using GPU device {}".format(self.gpu))
            self.model.cuda(device=self.gpu)

        self.criterion = nn.BCEWithLogitsLoss(size_average=True)
        self.optimiser = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.0)

    def train_model(self, epochs):
        for epoch in range(epochs):
            print("----------------")
            print("Epoch Number {}".format(epoch))
            for i, samples in enumerate(self.train_dataloader):
                print("Batch Number {}".format(i))
                inputs = samples['inputs'].float()
                labels = samples['labels'].float().unsqueeze(1)

                # Use GPU if available
                if self.use_GPU:
                    inputs = inputs.to(device=self.gpu)
                    labels = labels.to(device=self.gpu)
                y_hat = self.model.forward(inputs)
                loss = self.criterion(y_hat, labels)
                print("Loss -> {}".format(loss))

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                break
            print("Epoch complete -> computing current loss...")
            # Estimate training loss
            for i, samples in enumerate(self.train_eval_dataloader):
                inputs = samples['inputs'].float()
                labels = samples['labels'].float().unsqueeze(1)
                # Use GPU if available
                if self.use_GPU:
                    inputs = inputs.to(device=self.gpu)
                    labels = labels.to(device=self.gpu)
                y_hat = self.model.forward(inputs)
                loss = self.criterion(y_hat, labels)
                self.training_loss.append(loss)
                print("Training loss: {}".format(loss))
                break
            for i, samples in enumerate(self.val_dataloader):
                inputs = samples['inputs'].float()
                labels = samples['labels'].float().unsqueeze(1)
                # Use GPU if available
                if self.use_GPU:
                    inputs = inputs.to(device=self.gpu)
                    labels = labels.to(device=self.gpu)
                y_hat = self.model.forward(inputs)
                loss = self.criterion(y_hat, labels)
                self.validation_loss.append(loss)
                print("Validation loss: {}".format(loss))
            # Save model after each epoch
            torch.save(self.model, "./training_logs/unet.pt")
        print("Finished training -> saving losses")
        self.training_loss = np.save("./training_logs/training_loss", np.array(self.training_loss))
        self.validation_loss = np.save("./training_logs/validation_loss", np.array(self.validation_loss))


if __name__ == "__main__":
    unet = models.U_Net()
    trainer = Trainer(model=unet, training_path='./data/training_data/',
                      validation_path='./data/validation_data/', batch_size=16, train_eval_fraction=0.1, use_GPU=True)
    trainer.train_model(epochs=10)



