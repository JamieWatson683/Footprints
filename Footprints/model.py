import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class U_Net(nn.Module):
    def __init__(self, layer_number, filter_list, batch_norm_flag=True):
        super(U_Net, self).__init__()
        self.encoders = {}
        self.decoders = {}
        self.layer_number = layer_number
        self.filter_list = filter_list
        self.batch_norm_flag = batch_norm_flag

        # Put each layer object into dictionary containing
        for layer in range(1, layer_number):
            self.encoders[layer] = EncoderLayer(filter_list[layer-1], filter_list[layer])
            self.decoders[layer] = DecoderLayer(filter_list[layer_number - 1 - layer])


class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm_flag=True):
        super(EncoderLayer, self).__init__()
        self.batch_norm_flag = batch_norm_flag

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.batch_norm_flag:
            x = self.batchnorm1(x)
        self.featuremap = x
        x = self.pool1(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm_flag=True):
        super(DecoderLayer, self).__init__()
        self.batch_norm_flag = batch_norm_flag

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(out_channels, int(out_channels/2), stride=2)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip_data=None):
        if skip_data:
            x = torch.cat((x, skip_data), 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.batch_norm_flag:
            x = self.batchnorm1(x)
        x = self.deconv1(x)
        return x


class FinalDecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm_flag=True):
        super(FinalDecoderLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x, skip_data=None):
        if skip_data:
            x = torch.cat((x, skip_data), 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.batch_norm_flag:
            x = self.batchnorm1(x)
        x = self.conv3(x)
        self.probability = F.sigmoid(x)
        return x



