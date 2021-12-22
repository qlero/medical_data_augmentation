"""

This .py file contains the classes <CNN> and <classification>
adapted from the notebook getting_started found in the /examples/ 
subfolder in:
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
import medmnist
import torch
import torch.nn as nn
import torch.optim as optim

from medmnist import Evaluator

class CNN(nn.Module):
    """
    Defines a simple convolutional neural network with which to run 
    simple binary or multi-class classification.
    """
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()
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
    def __init__(self, n_channels, n_classes, task, learning_rate):
        # Defines the pipeline model
        self.model = CNN(in_channels=n_channels, num_classes=n_classes)
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
    def train(train_loader, epochs=3):
        """
        Runs the training phase for the loaded model for a given
        data loader and a given number of epochs
        """
        for epoch in range(epochs):
            train_correct = 0
            train_total = 0
            self.model.train()
            for inputs, targets in tqdm(train_loader):
                # Performs the forward + backward + optimize passes
                optimizer.zero_grad()
                outputs = self.model(inputs)
                if task == "multi-label, binary-class":
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
            print(f"Epoch {epoch} train. accuracy: {round(acc,2)}")
        # Records that the train phase was done
        self.training_done=True
    def test(test_loader, name_loader = "test"):
        """
        Runs the test phase for the trained model.
        """
        self.model.eval()
        test_correct = torch.tensor([])
        test_score = torch.tensor([])
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
            y_true = y_true.numpy()
            y_score = y_score.detach().numpy()
            evaluator = Evaluator(data_flag, split)
            metrics = evaluator.evaluate(y_score)
            print("== Evaluating model ==")
            print(f"{name_loader} -- accuracy: {round(metrics[0],2)}, ",
                  f"AUC: {round(metrics[1], 2)}")
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        