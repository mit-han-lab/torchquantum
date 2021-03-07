import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalConv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 2, 1)
        self.conv2 = nn.Conv2d(4, 4, 1, 1)
        self.conv3 = nn.Conv2d(4, 4, 2, 1)
        self.conv4 = nn.Conv2d(4, 4, 1, 1)
        self.conv5 = nn.Conv2d(4, 10, 2, 1)
        self.act = lambda x: x * x

    def forward(self, x):
        x = F.avg_pool2d(x, 6)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.act(x)
        x = self.conv5(x)
        output = F.log_softmax(x, dim=1)

        return output.squeeze()
