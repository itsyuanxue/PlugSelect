# 导入工具包
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader,Dataset

class AADNet(nn.Module):
    def __init__(self, classes_num, channel_num=32, embeddings_flag=False):
        super(EEGNet, self).__init__()
        self.drop_out = 0.25
        self.channel_num = channel_num
        self.embeddings_flag = embeddings_flag

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((self.channel_num-1, self.channel_num, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=self.channel_num,  # num_filters
                kernel_size=(1, self.channel_num*2),  # filter size
                bias=False
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(self.channel_num)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channel_num,  # input shape (8, C, T)
                out_channels=self.channel_num*2,  # num_filters
                kernel_size=(self.channel_num, 1),  # filter size 有问题  (32, 1)
                groups=self.channel_num,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(self.channel_num*2),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=self.channel_num*2,  # input shape (16, 1, T//4)
                out_channels=self.channel_num*2,  # num_filters
                kernel_size=(1, 16),  # filter size
                groups=self.channel_num*2,
                bias=False
            ),  # output shape (16, 1, T//
            nn.BatchNorm2d(self.channel_num*2),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.Conv2d(
                in_channels=self.channel_num*2,  # input shape (16, 1, T//4)
                out_channels=self.channel_num*2,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(self.channel_num*2),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.out = nn.Linear(14*channel_num, 64)   #0.1 64   0.5  64*7  1.0   64*15    1.5   64*23   2  64*31  20-280 15-210 10-140 5-70
        self.out2 = nn.Linear(64, classes_num)
        # self.out = nn.Linear(1232, classes_num)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.block_1(x)
        # print("block1", x.shape)
        x = self.block_2(x)
        # print("block2", x.shape)
        x = self.block_3(x)
        # print("block3", x.shape)
        x = x.view(x.size(0), -1)
        #x0 = x
        # print('jjjjjjj')
        # print(x.shape)
        x = self.out(x)
        x0=x
        x = self.out2(x)
        # return F.softmax(x, dim=1), x  # return x for visualization
        # return x0, x
        if self.embeddings_flag:
            return x, x0
        else:
            return x

class mydataset(Dataset):
    def __init__(self,x,y):
        super(mydataset).__init__()
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, item):
        return self.x[item],self.y[item]

class mydataset_bfe(Dataset):
    def __init__(self,x,y,adj):
        super(mydataset).__init__()
        self.x = x
        self.y = y
        self.adj = adj
    def __len__(self):
        return len( self.x)
    def __getitem__(self, item):
        return self.x[item],self.y[item],self.adj[item]
