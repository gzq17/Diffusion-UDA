import torch
import torch.nn as nn
import torch.nn.functional as F

class BnReluConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_size, padding, activate='relu', norm='batch', ndims=3):
        super(BnReluConv, self).__init__()
        if norm == 'batch':
            Norm_ = getattr(nn, 'BatchNorm%dd' % ndims)
            self.norm = Norm_(in_channels)
        else:
            Norm_ = getattr(nn, 'InstanceNorm%dd' % ndims)
            self.norm = Norm_(in_channels)
        if activate == 'relu':
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, padding, activate='relu', norm='batch', ndims=3):
        super(ResidualBlock, self).__init__()
        self.bn_relu_conv1 = BnReluConv(in_channels, out_channels, stride, kernel_size, padding, activate, norm, ndims)
        self.bn_relu_conv2 = BnReluConv(out_channels, out_channels, stride, kernel_size, padding, activate, norm, ndims)

    def forward(self, x):
        y = self.bn_relu_conv1(x)
        residual = y
        z = self.bn_relu_conv2(y)
        return z + residual

class DeResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_size, padding, activate='relu', norm='batch', ndims=3):
        super(DeResidualBlock, self).__init__()
        self.bn_relu_conv1 = BnReluConv(in_channels, out_channels, stride, kernel_size, padding, activate, norm, ndims)
        self.bn_relu_conv2 = BnReluConv(out_channels, out_channels, stride, kernel_size, padding, activate, norm, ndims)

    def forward(self, x1, x2):
        y = self.bn_relu_conv1(x1)
        y = self.bn_relu_conv2(y)
        return y + x2

class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, activate='relu', norm='batch', ndims=3):
        super(UpConv, self).__init__()
        Conv = getattr(nn, 'ConvTranspose%dd' % ndims)
        self.conv = Conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        if norm == 'batch':
            Norm_ = getattr(nn, 'BatchNorm%dd' % ndims)
            self.norm = Norm_(out_channels)
        else:
            Norm_ = getattr(nn, 'InstanceNorm%dd' % ndims)
            self.norm = Norm_(out_channels)
        if activate == 'relu':
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class SegmentationModel2D(nn.Module):

    def __init__(self, in_channel, out_channel, activate='leakrelu', norm='batch', num_list = [16, 32, 64, 128, 256], ndims=2):
        super(SegmentationModel2D, self).__init__()
        print(activate, norm)
        pool_ = getattr(nn, 'MaxPool%dd' % ndims)
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.conv1 = ResidualBlock(in_channel, out_channels=num_list[0], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=ndims)
        self.pool1 = pool_(kernel_size=2, stride=2)

        self.conv2 = ResidualBlock(num_list[0], num_list[1], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=ndims)
        self.pool2 = pool_(kernel_size=2, stride=2)

        self.conv3 = ResidualBlock(num_list[1], num_list[2], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=ndims)
        self.pool3 = pool_(kernel_size=2, stride=2)

        self.conv4 = ResidualBlock(num_list[2], num_list[3], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=ndims)
        self.pool4 = pool_(kernel_size=2, stride=2)

        self.conv5 = ResidualBlock(num_list[3], num_list[4], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=ndims)

        self.upconv1 = UpConv(num_list[4], num_list[3], activate=activate, norm=norm, ndims=ndims)
        self.deconv1 = DeResidualBlock(num_list[3] * 2, num_list[3], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=ndims)

        self.upconv2 = UpConv(num_list[3], num_list[2], activate=activate, norm=norm, ndims=ndims)
        self.deconv2 = DeResidualBlock(num_list[2] * 2, num_list[2], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=ndims)

        self.upconv3 = UpConv(num_list[2], num_list[1], activate=activate, norm=norm, ndims=ndims)
        self.deconv3 = DeResidualBlock(num_list[1] * 2, num_list[1], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=ndims)

        self.upconv4 = UpConv(num_list[1], num_list[0], activate=activate, norm=norm, ndims=ndims)
        self.deconv4 = DeResidualBlock(num_list[0] * 2, num_list[0], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=ndims)

        self.deconv5 = Conv(num_list[0], num_list[0], kernel_size=1, stride=1, bias=True)
        self.pred_prob = Conv(num_list[0], out_channel, kernel_size=1, stride=1, bias=True)
        self.pred_soft = nn.Sigmoid()

    def forward(self, x, fea=False):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)

        deconv1_1 = self.upconv1(conv5)
        concat_1 = torch.cat((deconv1_1, conv4), dim=1)
        deconv1_2 = self.deconv1(concat_1, deconv1_1)

        deconv2_1 = self.upconv2(deconv1_2)
        concat_2 = torch.cat((deconv2_1, conv3), dim=1)
        deconv2_2 = self.deconv2(concat_2, deconv2_1)

        deconv3_1 = self.upconv3(deconv2_2)
        concat_3 = torch.cat((deconv3_1, conv2), dim=1)
        deconv3_2 = self.deconv3(concat_3, deconv3_1)

        deconv4_1 = self.upconv4(deconv3_2)
        concat_4 = torch.cat((deconv4_1, conv1), dim=1)
        deconv4_2 = self.deconv4(concat_4, deconv4_1)

        deconv5_1 = self.deconv5(deconv4_2)
        pred_prob = self.pred_prob(deconv5_1)
        pred_soft = self.pred_soft(pred_prob)
        return pred_soft

