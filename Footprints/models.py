import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import transform, measure
import matplotlib.pyplot as plt


class U_Net(nn.Module):
    def __init__(self, input_depth=4):
        super(U_Net, self).__init__()

        # Architecture
        self.conv1 = EncoderLayer(in_channels=input_depth, out_channels=32)
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


class Big_U_Net(nn.Module):
    def __init__(self, input_depth=4):
        super(Big_U_Net, self).__init__()

        # Architecture
        self.conv1 = EncoderLayer(in_channels=input_depth, out_channels=64)
        self.conv2 = EncoderLayer(in_channels=64, out_channels=128)
        self.conv3 = EncoderLayer(in_channels=128, out_channels=256)
        self.conv4 = EncoderLayer(in_channels=256, out_channels=512)

        self.deconv1 = DecoderLayer(in_channels=512, out_channels=1024)
        self.deconv2 = DecoderLayer(in_channels=1024, out_channels=512)
        self.deconv3 = DecoderLayer(in_channels=512, out_channels=256)
        self.deconv4 = DecoderLayer(in_channels=256, out_channels=128)

        self.final = FinalDecoderLayer(in_channels=128, out_channels=64)

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


class Bayesian_Unet(nn.Module):
    def __init__(self, input_depth=4):
        super(Bayesian_Unet, self).__init__()

        # Architecture
        self.conv1 = EncoderLayer(in_channels=input_depth, out_channels=64, dropout_flag=False)
        self.conv2 = EncoderLayer(in_channels=64, out_channels=128, dropout_flag=False)
        self.conv3 = EncoderLayer(in_channels=128, out_channels=256, dropout_type='1D', dropout_prob=0.2)
        self.conv4 = EncoderLayer(in_channels=256, out_channels=512, dropout_type='1D', dropout_prob=0.2)

        self.deconv1 = DecoderLayer(in_channels=512, out_channels=1024, dropout_type='1D', dropout_prob=0.2)
        self.deconv2 = DecoderLayer(in_channels=1024, out_channels=512, dropout_type='1D', dropout_prob=0.2)
        self.deconv3 = DecoderLayer(in_channels=512, out_channels=256, dropout_flag=False)
        self.deconv4 = DecoderLayer(in_channels=256, out_channels=128, dropout_flag=False)

        self.final = FinalDecoderLayer(in_channels=128, out_channels=64)

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
    def __init__(self, in_channels, out_channels, batch_norm_flag=True, dropout_flag=True, dropout_type='2D',
                 dropout_prob=0.5):
        super(EncoderLayer, self).__init__()
        self.batch_norm_flag = batch_norm_flag
        self.dropout_flag = dropout_flag
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        if dropout_type == '1D':
            self.dropout = nn.Dropout(p=dropout_prob)
        elif dropout_type == '2D':
            self.dropout = nn.Dropout2d(p=dropout_prob)
        else:
            raise ValueError('Please set dropout type to either 1D or 2D!')

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm_flag:
            x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.batch_norm_flag:
            x = self.batchnorm2(x)
        x = F.relu(x)
        if self.dropout_flag:
            x = self.dropout(x)
        self.featuremap = x
        x = self.pool1(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm_flag=True, dropout_flag=True, dropout_type='2D',
                 dropout_prob=0.5):
        super(DecoderLayer, self).__init__()
        self.batch_norm_flag = batch_norm_flag
        self.dropout_flag = dropout_flag
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(out_channels, int(out_channels/2), kernel_size=2, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        if dropout_type == '1D':
            self.dropout = nn.Dropout(p=dropout_prob)
        elif dropout_type == '2D':
            self.dropout = nn.Dropout2d(p=dropout_prob)
        else:
            raise ValueError('Please set dropout type to either 1D or 2D!')

    def forward(self, x, skip_connection=False, skip_data=None):
        if skip_connection:
            x = torch.cat((x, skip_data), 1)
        x = self.conv1(x)
        if self.batch_norm_flag:
            x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.batch_norm_flag:
            x = self.batchnorm2(x)
        x = F.relu(x)
        if self.dropout_flag:
            x = self.dropout(x)
        x = self.deconv1(x)
        return x


class FinalDecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm_flag=True):
        super(FinalDecoderLayer, self).__init__()
        self.batch_norm_flag = batch_norm_flag

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x, skip_connection=False, skip_data=None):
        if skip_connection:
            x = torch.cat((x, skip_data), 1)
        x = self.conv1(x)
        if self.batch_norm_flag:
            x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.batch_norm_flag:
            x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.conv3(x)
        self.probability = F.sigmoid(x)
        return x


class Baseline(object):
    """Simply return the input mask as the prediction"""
    def __init__(self):
        pass

    def forward(self, inputs):
        return inputs[:,3,:,:]


class ResizeBaseline(object):
    """Resize mask and return as prediction"""
    def __init__(self, resizing):
        self.resizing = resizing

    def forward(self, inputs):
        mask = inputs[:,3,:,:].numpy().astype(int)
        prediction = np.zeros_like(mask)
        for index in range(len(mask)):
            maskbbox = measure.regionprops(mask[index])[0].bbox
            rescaled = transform.rescale(mask[index], scale=(self.resizing, 1.0))
            rescaled = ((rescaled / rescaled.max()) > 0.5).astype(int)
            rescaledbbox = measure.regionprops(rescaled)[0].bbox
            prediction[index, maskbbox[2]+rescaledbbox[0]-rescaledbbox[2]:maskbbox[2],
            rescaledbbox[1]:rescaledbbox[3]] = rescaled[rescaledbbox[0]:rescaledbbox[2],
                                               rescaledbbox[1]:rescaledbbox[3]]
        return prediction


