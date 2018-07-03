import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import models
import datasets


class Trainer(nn.Module):
    def __init__(self, model, training_path, validation_path, save_name='model', batch_size=8,
                 learning_rate=0.001, use_GPU=False):
        super(Trainer, self).__init__()
        print("Creating training session...")
        self.model = model
        self.save_name = save_name
        self.training_data = datasets.FootprintsDataset(training_path, augment=True)
        self.validation_data = datasets.FootprintsDataset(validation_path)
        self.use_GPU = use_GPU

        self.train_dataloader = DataLoader(self.training_data, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.validation_data, batch_size=batch_size)
        self.training_loss = []
        self.validation_loss = []
        self.batch_size = batch_size

        # Move model to GPU
        if self.use_GPU:
            self.gpu = torch.device('cuda:0')
            print("Using GPU device {}".format(self.gpu))
            self.model.cuda(device=self.gpu)

        self.criterion = nn.BCEWithLogitsLoss(size_average=True)
        self.eval_criterion = nn.BCEWithLogitsLoss(size_average=False)
        self.optimiser = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.0)

        print("Success")

    def train_model(self, epochs):
        for epoch in range(epochs):
            print("----------------")
            print("Epoch Number {}".format(epoch))
            for i, samples in enumerate(self.train_dataloader):

                inputs = samples['inputs'].float()
                labels = samples['labels'].float().unsqueeze(1)

                # Use GPU if available
                if self.use_GPU:
                    inputs = inputs.to(device=self.gpu)
                    labels = labels.to(device=self.gpu)
                y_hat = self.model.forward(inputs)
                loss = self.criterion(y_hat, labels)
                if i % 25 == 0:
                    print("Batch Number {}".format(i))
                    print("Loss -> {}".format(loss))
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                # del inputs, labels, loss, y_hat
                # torch.cuda.empty_cache()

            print("Epoch complete -> computing current loss...")
            # Estimate training loss
            with torch.no_grad():
                total_loss = 0
                for i, samples in enumerate(self.train_dataloader):
                    inputs = samples['inputs'].float()
                    labels = samples['labels'].float().unsqueeze(1)
                    # Use GPU if available
                    if self.use_GPU:
                        inputs = inputs.to(device=self.gpu)
                        labels = labels.to(device=self.gpu)
                    y_hat = self.model.forward(inputs)
                    loss = self.eval_criterion(y_hat, labels)
                    total_loss += loss
                    # del inputs, labels, loss, y_hat
                    # torch.cuda.empty_cache()
                total_loss = total_loss / (i*self.batch_size)
                self.training_loss.append(total_loss)
                print("Mean Training loss: {}".format(total_loss))

            with torch.no_grad():
                total_loss = 0
                for i, samples in enumerate(self.val_dataloader):
                    inputs = samples['inputs'].float()
                    labels = samples['labels'].float().unsqueeze(1)
                    # Use GPU if available
                    if self.use_GPU:
                        inputs = inputs.to(device=self.gpu)
                        labels = labels.to(device=self.gpu)
                    y_hat = self.model.forward(inputs)
                    loss = self.eval_criterion(y_hat, labels)
                    total_loss += loss
                    # del inputs, labels, loss, y_hat
                    # torch.cuda.empty_cache()
                total_loss = total_loss / (i*self.batch_size)
                self.validation_loss.append(total_loss)
                print("Validation loss: {}".format(total_loss))
            # Save model after each epoch
            print("Saving model")
            torch.save(self.model, "./training_logs/"+self.save_name+".pt")
            # Print loss summary so far
            print("Training loss over time:")
            for l in self.training_loss:
                print(l.cpu().numpy())
            print("Validation loss over time:")
            for l in self.validation_loss:
                print(l.cpu().numpy())
        print("Finished training -> saving losses")
        np.save("./training_logs/training_loss", np.array(self.training_loss))
        np.save("./training_logs/validation_loss", np.array(self.validation_loss))


if __name__ == "__main__":
    unet = models.U_Net()
    trainer = Trainer(model=unet, training_path='./data/training_data/',
                      validation_path='./data/validation_data/', batch_size=16, use_GPU=True)
    trainer.train_model(epochs=10)



