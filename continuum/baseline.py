import os
from typing import Iterable, Set, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

from continuum import ClassIncremental
from continuum.datasets import Core50
from continuum.tasks import split_train_val
from torchvision.transforms.transforms import Normalize, ToTensor

# Load the core50 data
core50 = Core50("core50/data/", train=True, download=False)


# A new classes scenario
scenario = ClassIncremental(
    core50,
    increment=5,
    initial_increment=10,
    transformations=[ ToTensor(), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
)

print(f"Number of classes: {scenario.nb_classes}.")
print(f"Number of tasks: {scenario.nb_tasks}.")

# Define a model
classifier = models.resnet18(pretrained=True)
classifier.fc = nn.Linear(512, 50)

if torch.cuda.is_available():
    print('cuda IS available')
    classifier.cuda()
else:
    print('cuda / GPU not available.')

# Tune the model hyperparameters
epochs = 1
lr = 0.01

# Define a loss function and criterion
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=lr)

# Iterate through our NC scenario
for task_id, train_taskset in enumerate(scenario):
    train_taskset, val_taskset = split_train_val(train_taskset, val_split=0.1)
    train_loader = DataLoader(train_taskset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_taskset, batch_size=32, shuffle=True)

    unq_cls_train = np.unique(train_taskset._y)
    unq_cls_validate = np.unique(val_taskset._y)
    print(f"This task contains {len(unq_cls_train)} unique classes")
    print(f"Train: {unq_cls_train}")
    print(f"Validate: {unq_cls_validate}")

    for epoch in range(epochs):

        print(f"<----- Epoch {epoch + 1} ------->")

        running_loss = 0.0
        for i, (x, y, t) in enumerate(train_loader):
            
            # Outputs batches of data, one scenario at a time
            x, y = x.cuda(), y.cuda()
            outputs = classifier(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0              

    print("Finished Training")

    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for x, y, t in val_loader:
            x, y = x.cuda(), y.cuda()
            outputs = classifier(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    print(f"Accuracy: {100.0 * correct / total}")

    