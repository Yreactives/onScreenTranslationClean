import torch
import torch.nn as nn



class CNN (nn.Module):
    def __init__(self, image_size = 32, num_classes = 2168):
        super(CNN, self).__init__()

        self.filter_size = 32
        self.image_size = image_size // 4
        self.hidden_size = 128
        self.conv1 = nn.Conv2d(1, self.filter_size, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.filter_size, self.filter_size*2, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.filter_size * 2 * self.image_size * self.image_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, num_classes)
    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(output)
        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)

        output = self.fc1(output)
        output = self.fc2(output)
        return output

class CNNwithBatchNorm (nn.Module):
    def __init__(self, image_size = 32, num_classes = 2168):
        super(CNNwithBatchNorm, self).__init__()

        self.filter_size = 64
        self.image_size = image_size // 4
        self.hidden_size = 256
        self.conv1 = nn.Conv2d(1, self.filter_size, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.filter_size)
        self.conv2 = nn.Conv2d(self.filter_size, self.filter_size*2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(self.filter_size*2)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.filter_size * 2 * self.image_size * self.image_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, num_classes)
    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.pool(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)
        output = self.pool(output)

        output = self.fc1(output)
        output = self.fc2(output)
        return output