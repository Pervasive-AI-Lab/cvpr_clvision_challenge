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


######################### UTIL FUNCTIONS ################################

# Workaround until we add core50_train.csv to the data folder (I don't have permissions)
# Borrowed from continuum source code
def train_img_ids(csv_file="core50_train.csv"):
    train_image_ids = set()
    with open(csv_file, 'r') as f:
        for line in f:
            image_id = line.split(",")[0].split(".")[0]
            train_image_ids.add(image_id)
    return train_image_ids

# Our directory structure is a little different, so overriding get_data
class MyCore50(Core50):

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the CORe50 data.

        CORe50, in one of its many iterations, is made of 50 objects, each present
        in 10 different domains (in-door, street, garden, etc.).

        In class incremental (NC) setting, those domains won't matter.

        In instance incremental (NI) setting, the domains come one after the other,
        but all classes are present since the first task. Seven domains are allocated
        for the train set, while 3 domains are allocated for the test set.

        In the case of the test set, all domains have the "dummy" label of 0. The
        authors designed this dataset with a fixed test dataset in mind.
        """
        x, y, t = [], [], []

        domain_counter = 0
        for domain_id in range(10):
            # We walk through the 10 available domains.
            domain_folder = os.path.join(self.data_path, f"s{domain_id + 1}")

            has_images = False
            for object_id in range(50):
                # We walk through the 50 available object categories.
                object_folder = os.path.join(domain_folder, f"o{object_id + 1}")

                for path in os.listdir(object_folder):
                    image_id = path.split(".")[0]

                    if (
                        (self.train and image_id not in self.train_image_ids) or  # type: ignore
                        (not self.train and image_id in self.train_image_ids)  # type: ignore
                    ):
                        continue

                    x.append(os.path.join(object_folder, path))
                    y.append(object_id)
                    if self.train:  # We add a new domain id for the train set.
                        t.append(domain_counter)
                    else:  # Test set is fixed, therefore we artificially give a unique domain.
                        t.append(0)

                    has_images = True
            if has_images:
                domain_counter += 1

        x = np.array(x)
        y = np.array(y)
        t = np.array(t)

        return x, y, t


####################### CONTINUOUS LEARNING IMPLEMENTATION ##############################

# Load the core50 data
core50 = MyCore50("core50/data/", train=True, download=False)

# A new classes scenario
scenario = ClassIncremental(
    core50,
    increment=5,
    initial_increment=10
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

    