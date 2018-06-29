import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class U_Net(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()

        # Architecture
        self.conv1 = EncoderLayer(in_channels=4, out_channels=32)
        self.conv2 = EncoderLayer(in_channels=32, out_channels=64)
        self.conv3 = EncoderLayer(in_channels=64, out_channels=128)
        self.conv4 = EncoderLayer(in_channels=128, out_channels=256)

        self.deconv1 = DecoderLayer(in_channels=256, out_channels=512)
        self.deconv2 = DecoderLayer(in_channels=512, out_channels=256)
        self.deconv3 = DecoderLayer(in_channels=256, out_channels=128)
        self.deconv4 = DecoderLayer(in_channels=128, out_channels=64)

        self.final = FinalDecoderLayer(in_channels=64, out_channels=32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.deconv1(x)
        x = self.deconv2(x, skip_connection=True, skip_data=self.conv4.featuremap)
        x = self.deconv3(x, skip_connection=True, skip_data=self.conv3.featuremap)
        x = self.deconv4(x, skip_connection=True, skip_data=self.conv2.featuremap)

        x = self.final(x, skip_connection=True, skip_data=self.conv1.featuremap)

        return x



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
        self.deconv1 = nn.ConvTranspose2d(out_channels, int(out_channels/2), kernel_size=2, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip_connection=False, skip_data=None):
        if skip_connection:
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
        self.batch_norm_flag = batch_norm_flag

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x, skip_connection=False, skip_data=None):
        if skip_connection:
            x = torch.cat((x, skip_data), 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.batch_norm_flag:
            x = self.batchnorm1(x)
        x = self.conv3(x)
        self.probability = F.sigmoid(x)
        return x



