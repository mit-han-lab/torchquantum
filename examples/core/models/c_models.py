import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalConv0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class ClassicalConv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.avg_pool2d(x, 6)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class ClassicalConv1Resize4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


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


class ClassicalFC0(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        x = self.fc1(x)[:, :10]
        x = x * x

        output = F.log_softmax(x, dim=1)

        return output.squeeze()


class ClassicalFC1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        bsz = x.shape[0]
        x = x.view(bsz, 784)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x * x

        output = F.log_softmax(x, dim=1)

        return output.squeeze()


model_dict = {
    'c_conv0': ClassicalConv0,
    'c_conv1': ClassicalConv1,
    'c_conv1_resize4': ClassicalConv1Resize4,
    'c_conv2': ClassicalConv2,
    'c_fc0': ClassicalFC0,
    'c_fc1': ClassicalFC1
}
