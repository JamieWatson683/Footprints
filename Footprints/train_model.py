import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import models
import datasets


class Trainer(nn.Module):
    def __init__(self, model, training_path, validation_path, save_name='model', batch_size=32, eval_batch_size=64,
                 learning_rate=0.001, use_GPU=False):
        super(Trainer, self).__init__()
        print("Creating training session...")
        self.model = model
        self.save_name = save_name
        self.training_data = datasets.FootprintsDataset(training_path, augment=True)
        self.training_eval_data = datasets.FootprintsDataset(training_path, augment=False)
        self.validation_data = datasets.FootprintsDataset(validation_path, augment=False)
        self.use_GPU = use_GPU

        self.train_dataloader = DataLoader(self.training_data, batch_size=batch_size, shuffle=True)
        self.train_eval_dataloader = DataLoader(self.training_eval_data, batch_size=eval_batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.validation_data, batch_size=eval_batch_size)
        self.training_loss = []
        self.validation_loss = []
        self.training_iou = []
        self.validation_iou = []
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
                if i % 10 == 0:
                    print("Batch Number {}".format(i))
                    print("Loss -> {}".format(loss))
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()


            print("Epoch complete -> computing current loss and IoU...")
            # Estimate training loss
            with torch.no_grad():
                total_loss = 0
                iou_all = 0
                for i, samples in enumerate(self.train_eval_dataloader):
                    inputs = samples['inputs'].float()
                    labels = samples['labels'].float().unsqueeze(1)
                    # Use GPU if available
                    if self.use_GPU:
                        inputs = inputs.to(device=self.gpu)
                        labels = labels.to(device=self.gpu)
                    y_hat = self.model.forward(inputs)
                    loss = self.eval_criterion(y_hat, labels)
                    total_loss += loss
                    # IoU
                    predictions = self.model.final.probability > 0.5
                    iou = self.IoU(predictions, labels)
                    iou_all += iou

                total_loss = total_loss / len(self.training_eval_data)
                iou_all = iou_all / len(self.training_eval_data)
                self.training_loss.append(total_loss)
                self.training_iou.append(iou_all)
                print("Mean Training loss: {}".format(total_loss))
                print("Mean Training IoU: {}".format(iou_all))

            with torch.no_grad():
                total_loss = 0
                iou_all = 0
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
                    # IoU
                    predictions = self.model.final.probability > 0.5
                    iou = self.IoU(predictions, labels)
                    iou_all += iou

                total_loss = total_loss / len(self.validation_data)
                iou_all = iou_all / len(self.validation_data)
                self.validation_loss.append(total_loss)
                self.validation_iou.append(iou_all)
                print("Mean Validation loss: {}".format(total_loss))
                print("Mean Validation IoU: {}".format(iou_all))
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
        np.save("./training_logs/training_iou", np.array(self.training_iou))
        np.save("./training_logs/validation_iou", np.array(self.validation_iou))

    def IoU(self, predictions, labels):
        intersection = np.multiply(predictions, labels).sum(-1).sum(-1)
        union = np.maximum(predictions, labels).sum(-1).sum(-1)
        return (intersection / union).sum()


if __name__ == "__main__":
    unet = models.U_Net()
    trainer = Trainer(model=unet, training_path='./data/training_data/',
                      validation_path='./data/validation_data/', batch_size=32, eval_batch_size=128, use_GPU=True)
    trainer.train_model(epochs=25)



