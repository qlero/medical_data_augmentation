"""

This .py file contains the classes <CNN> and <classification>, and the
function <run_classifier_pipeline> inspired in part from the notebook 
getting_started.ipynb found in the /examples/ subfolder in:
    > https://github.com/MedMNIST/MedMNIST/

Citations:
    > Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, 
    > Hanspeter Pfister, Bingbing Ni. "MedMNIST v2: A Large-Scale Lightweight 
    > Benchmark for 2D and 3D Biomedical Image Classification". arXiv preprint 
    > arXiv:2110.14795, 2021.
    > 
    > Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: 
    > A Lightweight AutoML Benchmark for Medical Image Analysis". IEEE 18th 
    > International Symposium on Biomedical Imaging (ISBI), 2021.

"""

##############################
########## IMPORTS ###########
##############################

import json
import matplotlib.pyplot as plt
import medmnist
import torch
import torch.nn as nn
import torch.optim as optim

from medmnist import Evaluator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

##############################
########## CLASSES ###########
##############################

class CNN(nn.Module):
    """
    Defines a simple convolutional neural network with which to run 
    simple binary or multi-class classification.
    """
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class classification():
    """
    Defines a machine learning classification pipeline using
    the CNN network previously declared.
    """
    def __init__(self, n_channels, n_classes, task, learning_rate, name):
        # Defines the pipeline model
        self.model = CNN(in_channels=n_channels, num_classes=n_classes)
        self.data_flag = name
        # Defines the loss function and optimizer
        if task == "multi-label, binary-class": 
            self.criterion = nn.BCEWithLogitsLoss()
        else: 
            self.criterion = nn.CrossEntropyLoss()
        self.task = task
        self.optimizer = optim.SGD(self. model.parameters(), 
                                   lr=learning_rate, 
                                   momentum=0.9)
        # Creates a boolean check to avoid testing before training
        self.training_done=False
        # Defines a list holder for the epoch accuracies for each
        # epoch of a training process
        self.accuracy_per_epoch = []
    def train(self, train_loader, val_loader, epochs=3):
        """
        Runs the training phase for the loaded model for a given
        data loader and a given number of epochs
        """
        for epoch in range(epochs):
            print(f"===================\nEpoch {epoch}\n")
            train_correct = 0
            train_total = 0
            self.model.train()
            for inputs, targets in tqdm(train_loader):
                # Performs the forward + backward + optimize passes
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                if self.task == "multi-label, binary-class":
                    targets = targets.to(torch.float32)
                else:
                    targets = targets.squeeze().long()
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                # Computes the batch results
                train_correct += sum(targets == torch.argmax(outputs, 1))
                train_total += len(outputs)
            # Computes the epoch accuracies
            acc = train_correct.item()/train_total
            self.accuracy_per_epoch.append(acc)
            print(f"train -- accuracy: {round(acc,2)}")
            self.test(val_loader, split = "val")
        # Records that the train phase was done
        self.training_done=True
        print("===================")
    def test(self, test_loader, label_names = None, split = "test",
             display_confusion_matrix=False):
        """
        Runs the test phase for the trained model.
        """
        self.model.eval()
        y_true = torch.tensor([])
        y_score = torch.tensor([])
        # Computes for the whole given data loader, the corresponding
        # accuracy with the previously trained model.
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self.model(inputs)
                if self.task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    outputs = outputs.softmax(dim=-1)
                else:
                    targets = targets.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().resize_(len(targets), 1)
                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)
            y_preds = torch.argmax(y_score, 1)
            y_true = y_true.numpy()
            y_score = y_score.detach().numpy()
            evaluator = Evaluator(self.data_flag, split)
            metrics = evaluator.evaluate(y_score)
        if display_confusion_matrix:
            print(f"{split} -- accuracy: {round(metrics[0],2)}, ",
                  f"AUC: {round(metrics[1], 2)}")
            cm = confusion_matrix(y_true.tolist(), 
                                  y_preds.tolist())
            if label_names is None:
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm
                )
            else:
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=label_names
                )
            plt.figure(figsize=(10,10))
            disp.plot()
            plt.xticks(rotation=90)
            plt.show()
        else:
            print(f"{split} -- accuracy: {round(metrics[0],2)}, ",
                  f"AUC: {round(metrics[1], 2)}")

##############################
######### FUNCTIONS ##########
##############################
            
def run_classifier_pipeline(name, info_flags, imported_data,
                            learning_rate=0.01, epochs=5):
    """
    Runs the training and testing process for the classifier 
    declared above.
    """
    # Prints the description of the dataset
    info = info_flags[name][0]
    print(json.dumps(info, sort_keys=False, indent=2))
    # Declares the model
    clf = classification(
        n_channels=info["n_channels"],
        n_classes=len(info["label"]),
        task=len(info["task"]),
        learning_rate=learning_rate, 
        name=name
    )
    # Runs the training phase
    clf.train(
        train_loader=imported_data[3], 
        val_loader=imported_data[5], 
        epochs=epochs
    )
    # Runs the testing phase
    clf.test(
        test_loader=imported_data[4], 
        label_names=info["label"].values(), 
        display_confusion_matrix=True
    )
    return clf