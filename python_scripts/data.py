"""

This .py file contains the functions <retrieve_flag_info> and
<create_data_loaders> adapted from the notebook getting_started
found in the /examples/ subfolder in:
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
import torchvision.transforms as transforms
import torch.utils.data as data

from IPython.display import Image 
from medmnist import INFO, Evaluator

def retrieve_flag_info(flag): 
    """
    Given a MedMNIST flag string, retrieves the related
    info, task, n_channels, n_classes, and DataClass.
    """
    info = INFO[flag]
    task = info["task"]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])
    return (info, task, n_channels, n_classes, DataClass)

def create_data_loaders(DataClass, batch_size=128, download=True):
    """
    Creates the train, train_at_eval (validation), and test 
    data loaders of a given dataclass from the MedMNIST and returns
    them as a tuple.
    """
    # Declares the dataloader pre-processing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    # Loads the data
    train_dataset = DataClass(split="train", 
                              transform=data_transform, 
                              download=download)
    test_dataset = DataClass(split="test", 
                             transform=data_transform, 
                             download=download)
    pil_dataset = DataClass(split="train", 
                            download=download)
    # Encapsulates data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, 
                                   batch_size=batch_size, 
                                   shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                           batch_size=2*batch_size, 
                                           shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, 
                                  batch_size=2*batch_size, 
                                  shuffle=False)
    # Prints the resulting loader metadata
    print("==================="); print(train_dataset)
    print("==================="); print(test_dataset)
    print("===================")
    # Visualizes the imported data
    display(train_dataset.montage(length=10))
    return train_loader, train_loader_at_eval, test_loader