import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.first_conv1d = nn.Conv1d(k,64,1)
        self.second_conv1d = nn.Conv1d(64,128,1)
        self.third_conv1d = nn.Conv1d(128,1024,1)
        self.first_layer = nn.Linear(1024,512)
        self.second_layer = nn.Linear(512,256)
        self.third_layer = nn.Linear(256,k*k)
        self.first_batch_norm = nn.BatchNorm1d(64)
        self.second_batch_norm = nn.BatchNorm1d(128)
        self.third_batch_norm = nn.BatchNorm1d(1024)
        self.fourth_batch_norm = nn.BatchNorm1d(512)
        self.fifth_batch_norm = nn.BatchNorm1d(256)

    def forward(self, input):
        temp_1d = self.first_conv1d(input)
        relu_first = F.relu(self.first_batch_norm(temp_1d))
        temp_2d = self.second_conv1d(relu_first)
        relu_second = F.relu(self.second_batch_norm(temp_2d))
        temp_3d = self.third_conv1d(relu_second)
        relu_third = F.relu(self.third_batch_norm(temp_3d))
        size_relu_third = relu_third.size(-1)
        max_pool = nn.MaxPool1d(size_relu_third)(relu_third)
        flatten = nn.Flatten(1)(max_pool)
        relu_fourth = F.relu(self.fourth_batch_norm(self.first_layer(flatten)))
        relu_fifth = F.relu(self.fifth_batch_norm(self.second_layer(relu_fourth)))
        start_ocurrence = torch.eye(self.k, requires_grad=True).repeat(input.size(0), 1, 1)
        if relu_fifth.is_cuda: start_ocurrence = start_ocurrence.cuda()
        fin_matrix = self.third_layer(relu_fifth).reshape(-1, self.k, self.k) + start_ocurrence
        return fin_matrix

class TransformBlock(nn.Module):
  def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.first_conv1d = nn.Conv1d(3,64,1)
        self.conv1d_l = nn.Conv1d(64, 64, 1)
        self.second_conv1d = nn.Conv1d(64,128,1)
        self.first_resblock = ResidualBlock(128, 256)
        self.second_resblock = ResidualBlock(256, 512)
        self.third_resblock = ResidualBlock(512, 1024)
        self.third_conv1d = nn.Conv1d(1024,1024,1)
        self.first_batch_norm = nn.BatchNorm1d(64)
        self.second_batch_norm = nn.BatchNorm1d(128)
        self.third_batch_norm = nn.BatchNorm1d(1024)

  def forward(self, input):
        matrix3x3 = self.input_transform(input)
        input_tran = torch.transpose(input, 1, 2)
        input_bmm = torch.transpose(torch.bmm(input_tran, matrix3x3), 1, 2)
        relu_first_batch = F.relu(self.first_batch_norm(self.first_conv1d(input_bmm)))
        init_copy = copy.copy(relu_first_batch)
        conv_layer1 = self.conv1d_l(self.conv1d_l(relu_first_batch)) + init_copy
        second_copy = copy.copy(conv_layer1)
        matrix64x64 = self.feature_transform(conv_layer1)
        conv_tran = torch.transpose(conv_layer1, 1, 2)
        relu_second_batch = F.relu(self.second_batch_norm(self.second_conv1d(self.conv1d_l(torch.transpose(torch.bmm(conv_tran, matrix64x64), 1, 2)) + second_copy)))
        third_layer = self.third_resblock(self.second_resblock(self.first_resblock(relu_second_batch)))
        third_batch_norm = self.third_batch_norm(self.third_conv1d(third_layer))
        maxpool = nn.MaxPool1d(third_batch_norm.size(-1))(third_batch_norm)
        out = nn.Flatten(1)(maxpool)
        return out, matrix3x3, matrix64x64

class PointResNet(nn.Module):
    def __init__(self, classes = 40):
        super().__init__()
        self.transform = TransformBlock()
        self.first_layer = nn.Linear(1024, 512)
        self.second_layer = nn.Linear(512, 256)
        self.third_layer = nn.Linear(256, classes)
        self.first_batch_norm = nn.BatchNorm1d(512)
        self.second_batch_norm = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        input = input.transpose(1,2).contiguous()
        transform, matrix3x3, matrix64x64 = self.transform(input)
        first_relu = F.relu(self.first_batch_norm(self.first_layer(transform)))
        second_relu = F.relu(self.second_batch_norm(self.dropout(self.second_layer(first_relu))))
        out = self.third_layer(second_relu)
        return self.logsoftmax(out) # not sure if this is needed or not

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.first_conv1d = nn.Conv1d(in_channels, out_channels, 1)
        self.first_batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.second_conv1d = nn.Conv1d(out_channels, out_channels, 1)
        self.second_batch_norm = nn.BatchNorm1d(out_channels)
        self.sequential = nn.Sequential()
        if in_channels != out_channels:
            self.sequential.add_module("conv", nn.Conv1d(in_channels, out_channels, 1))
            self.sequential.add_module("bn", nn.BatchNorm1d(out_channels))

    def forward(self, x):
        first_layer = self.first_conv1d(x)
        second_layer = self.first_batch_norm(first_layer)
        relu_layer = self.relu(second_layer)
        third_layer = self.second_conv1d(relu_layer)
        batch_norm_layer = self.second_batch_norm(third_layer)
        batch_norm_layer += self.sequential(x)
        fin_relu = self.relu(batch_norm_layer)
        return fin_relu