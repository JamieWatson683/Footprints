import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import models
import datasets


class Trainer(nn.Module):
    def __init__(self, model, training_input_path, training_label_path, batch_size=8):
        super(Trainer, self).__init__()
        self.model = model
        self.training_data = datasets.FootprintsDataset(training_input_path, training_label_path)
        self.dataloader = DataLoader(self.training_data, batch_size=batch_size, shuffle=True)

        self.criterion = nn.BCEWithLogitsLoss(size_average=False)
        self.optimiser = torch.optim.RMSprop(model.parameters(), lr=0.001, momentum=0.0)

    def train_model(self, epochs):
        for epoch in range(epochs):
            epoch_loss = 0
            for i, samples in enumerate(self.dataloader):
                print("Batch Number {}".format(i))
                inputs = samples['inputs'].float()
                labels = samples['labels'].float()

                y_hat = self.model.forward(inputs)
                print(inputs.shape)
                print(labels.shape)
                loss = self.criterion(y_hat, labels)
                epoch_loss += loss

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
        print("Epoch complete -> training loss: {}".format(epoch_loss))


if __name__ == "__main__":
    unet = models.U_Net()
    train = Trainer(model=unet, training_input_path="./data/training_data/inputs/",
                    training_label_path="./data/training_data/labels/")
    train.train_model(1)



