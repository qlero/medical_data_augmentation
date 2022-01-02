"""

This .py file contains the functions <create_data_loaders>, 
<display_set_statistics>, <import_dataset>, <retrieve_flag_info> 
adapted from the notebook getting_started found in the /examples/ 
subfolder in:
    > https://github.com/MedMNIST/MedMNIST/
    
Objects are declared in alphabetical order.

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

import gc
import matplotlib.pyplot as plt
import medmnist
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .variational_autoencoders import ConditionalVAE, JointVAE, one_hot
from collections import Counter
from IPython.display import Image 
from medmnist import INFO, Evaluator
from torchvision.transforms import Lambda

from torchfusion.gan.learners import *
from torchfusion.gan.applications import StandardGenerator, StandardProjectionDiscriminator
from torch.optim import Adam
from torchfusion.datasets import fashionmnist_loader
import torch.cuda as cuda
import torch.nn as nn
from torch.distributions import Normal

##############################
########## CLASSES ###########
##############################

class MyDataset(data.Dataset):
    """
    Custom Dataset class to allow for data transforms.
    """
    def __init__(self, X, y, transform=None):
        self.data = X
        self.target = y
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        # Normalize your data here
        if self.transform:
            x = self.transform(x)

        return x, y
    
    def __len__(self):
        return len(self.data)

##############################
######### FUNCTIONS ##########
##############################

def check_cuda_availability():
    """
    Checks whether CUDA is available.
    """
    print(
        torch.cuda.is_available(),",",
        torch.cuda.current_device(),",",
        torch.cuda.device(0),",",
        torch.cuda.device_count(),"\n",
        torch.cuda.get_device_name(0),
        sep=""
    )

def create_data_loaders(DataClass, validation_split=0.1,
                        batch_size=128, upscale=False, download=True):
    """
    Creates the train, train_at_eval (validation), and test 
    data loaders of a given dataclass from the MedMNIST and returns
    them as a tuple.
    Source for WRS:
        > https://opensourcebiology.eu/2021/11/19/
        >   using-weighted-random-sampler-in-pytorch/
    """
    # Declares the dataloader pre-processing
    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    if not upscale:
        data_transform_upscale = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    else:
        data_transform_upscale = transforms.Compose([
            transforms.ToTensor(),
            transforms.Scale((32,32)),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    # Loads the data
    train_dataset = DataClass(split="train", 
                              transform=data_transform_upscale, 
                              download=download)
    val_dataset = DataClass(split="val", 
                              transform=data_transform, 
                              download=download)
    test_dataset = DataClass(split="test", 
                             transform=data_transform, 
                             download=download)
    # Visualizes the imported data
    print("===================")
    print("Montage of randomly extracted images from the dataset:")
    display(train_dataset.montage(length=10))
    # Constructs the Weighted Random Sampler for the training dataset
    y_train = [x[0] for x in train_dataset.labels]
    weight = Counter(y_train)
    weight = {key:1/value for key,value in weight.items()}
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = data.WeightedRandomSampler(
        samples_weight.type('torch.DoubleTensor'), 
        len(samples_weight)
    )
    # Encapsulates data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, 
                                   batch_size=batch_size, 
                                   sampler=sampler)
    val_loader = data.DataLoader(dataset=val_dataset,
                                         batch_size=2*batch_size, 
                                         shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, 
                                  batch_size=2*batch_size, 
                                  shuffle=False)
    # Prints the resulting loader metadata
    print("===================")
    ret = (train_dataset, test_dataset, val_dataset, 
           train_loader, test_loader, val_loader)
    return ret

def display_set_statistics(datasets, dataset_info, name):
    """
    Given a Pytorch dataset object and its information object
    Computes the histogram of sample splits between the different
    dataset classes
    """
    true_labels_train = [entry[1][0] for entry in datasets[0]]
    true_labels_validation = [entry[1][0] for entry in datasets[1]]
    true_labels_test = [entry[1][0] for entry in datasets[2]]
    label_names = dataset_info[0]["label"]
    dataset_counter_train = Counter(true_labels_train).items()
    dataset_counter_validation = Counter(true_labels_validation).items()
    dataset_counter_test = Counter(true_labels_test).items()
    statistics_train = {label_names[f"{key}"]:value 
                        for key,value in dataset_counter_train}
    statistics_validation = {label_names[f"{key}"]:value 
                             for key,value in dataset_counter_validation}
    statistics_test = {label_names[f"{key}"]:value 
                       for key,value in dataset_counter_test}
    plt.figure(figsize=(15,8))   
    plt.subplot(1, 3, 1)
    plt.bar(statistics_train.keys(), statistics_train.values())
    plt.title(f"Label distribution in\nthe {name} training set")
    plt.xticks(rotation=90)
    plt.subplot(1, 3, 2)
    plt.bar(statistics_validation.keys(), statistics_validation.values())
    plt.title(f"Label distribution in\nthe {name} validation set")
    plt.xticks(rotation=90)
    plt.subplot(1, 3, 3)
    plt.bar(statistics_test.keys(), statistics_test.values())
    plt.title(f"Label distribution in\nthe {name} test set")
    plt.xticks(rotation=90)
    plt.show()

def generate_augmented_dataset_condVAE(n_channels, n_classes, latent_dims, 
                               model_path, 
                               original_train_set, batch_size,
                               test_loader, val_loader, n_sampling=None,
                               weighted_sampling=False):
    """
    Generates an augmented training set for a classifier task.
    """
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    # Loads conditional VAE model
    with torch.no_grad():
        model = ConditionalVAE(n_channels, n_classes, latent_dims).cuda()
        model.load_state_dict(torch.load(model_path))
        # Generates new images with their labels
        if weighted_sampling:
            classes_counter = Counter([x[0] for x in original_train_set.labels])
            max_nb_elements = max(classes_counter.values())
            weighted_counter = {k:(abs(v-max_nb_elements)+100) 
                                for k,v in classes_counter.items()}
            weighted_counter = {k:v/sum(weighted_counter.values()) 
                                for k,v in weighted_counter.items()}
            labels = np.random.choice(list(range(n_classes)),
                                 n_sampling,
                                 p=list(weighted_counter.values()))
        else:
            labels = np.random.randint(0, n_classes, n_sampling)
        images = model.sample(
            n_sampling,
            one_hot(torch.tensor(labels).int(),n_classes).cuda()
        ).detach().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # Generates the augmented dataset
    #for entry in original_train_set:
    old_labels = np.array([x[0] for x in original_train_set.labels])
    new_dataset = data.TensorDataset(images, torch.Tensor(labels).int())
    old_dataset = MyDataset(original_train_set.imgs, 
                            torch.Tensor(old_labels).int(), 
                            transform=data_transform)
    new_dataset = data.ConcatDataset([new_dataset, old_dataset])
    # Creates the Weighted Random Sampler
    targets = []
    for _, t in new_dataset:
        targets.append(t)
    targets = torch.tensor(targets)
    class_sample_count = torch.tensor([(targets == t).sum() 
                                       for t in torch.unique(targets, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in targets])
    sampler = data.WeightedRandomSampler(samples_weight, len(samples_weight))
    # Encapsulates data into dataloader form
    train_loader = data.DataLoader(
        dataset=new_dataset,
        batch_size=batch_size,
        sampler=sampler#shuffle=True
    )
    return new_dataset, "", "", train_loader, test_loader, val_loader

def generate_augmented_dataset_condGAN(n_channels, n_classes, latent_dims, 
                               model_path, 
                               original_train_set, batch_size,
                               test_loader, val_loader, n_sampling=None,
                               weighted_sampling=False):
    """
    Generates an augmented training set for a classifier task.
    """
    data_transform_old = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    data_transform_new = transforms.Compose([
        transforms.Scale((28,28))
    ])
    # Loads conditional GAN model
    G = StandardGenerator(
        output_size=(n_channels,32,32),
        latent_size=latent_dims,
        num_classes=n_classes
    )
    if cuda.is_available():
        G = nn.DataParallel(G.cuda())
    learner = RAvgHingeGanLearner(G,None)
    learner.load_generator(model_path)
    # Generates new images with their labels
    if weighted_sampling:
        classes_counter = Counter([x[0] for x in original_train_set.labels])
        max_nb_elements = max(classes_counter.values())
        weighted_counter = {k:(abs(v-max_nb_elements)+100) 
                            for k,v in classes_counter.items()}
        weighted_counter = {k:v/sum(weighted_counter.values()) 
                            for k,v in weighted_counter.items()}
        labels = np.random.choice(list(range(n_classes)),
                             n_sampling,
                             p=list(weighted_counter.values()))
    else:
        labels = np.random.randint(0, n_classes, n_sampling)
    labels = torch.Tensor(labels).int()
    dist = Normal(0,1)
    latent_vectors = dist.sample((len(labels),latent_dims))
    images = learner.predict([latent_vectors, labels]).detach().cpu()
    del G
    del learner
    gc.collect()
    torch.cuda.empty_cache()
    # Generates the augmented dataset
    #for entry in original_train_set:
    new_dataset = MyDataset(images, 
                            labels, 
                            transform=data_transform_new)
    old_labels = np.array([x[0] for x in original_train_set.labels])
    old_dataset = MyDataset(original_train_set.imgs, 
                            torch.Tensor(old_labels).int(), 
                            transform=data_transform_old)
    new_dataset = data.ConcatDataset([new_dataset, old_dataset])
    # Creates the Weighted Random Sampler
    targets = []
    for _, t in new_dataset:
        targets.append(t)
    targets = torch.tensor(targets)
    class_sample_count = torch.tensor([(targets == t).sum() 
                                       for t in torch.unique(targets, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in targets])
    sampler = data.WeightedRandomSampler(samples_weight, len(samples_weight))
    # Encapsulates data into dataloader form
    train_loader = data.DataLoader(
        dataset=new_dataset,
        batch_size=batch_size,
        sampler=sampler#shuffle=True
    )
    return new_dataset, "", "", train_loader, test_loader, val_loader

def generate_augmented_dataset_jointVAE(n_channels, latent_dims, categorical_dims,
                               model_path, 
                               original_train_set, batch_size,
                               test_loader, val_loader, n_sampling=None):
    """
    Generates an augmented training set for a classifier task.
    """
    # Loads conditional VAE model
    with torch.no_grad():
        model = JointVAE(n_channels, latent_dims, categorical_dims).cuda()
        model.load_state_dict(torch.load(model_path))
    data_transform_new = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
        Lambda(lambda img: model.generate(torch.stack([img, img]).cuda())[0].detach().cpu())
    ])
    data_transform_old = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])
    # Generates the augmented dataset
    #for entry in original_train_set:
    old_labels = np.array([x[0] for x in original_train_set.labels])
    new_dataset = MyDataset(original_train_set.imgs, 
                            torch.Tensor(old_labels).int(), 
                            transform=data_transform_new)
    old_dataset = MyDataset(original_train_set.imgs, 
                            torch.Tensor(old_labels).int(), 
                            transform=data_transform_old)
    new_dataset = data.ConcatDataset([new_dataset, old_dataset])
    # Creates the Weighted Random Sampler
    targets = []
    for _, t in new_dataset:
        targets.append(t)
    targets = torch.tensor(targets)
    class_sample_count = torch.tensor([(targets == t).sum() 
                                       for t in torch.unique(targets, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in targets])
    sampler = data.WeightedRandomSampler(samples_weight, len(samples_weight))
    # Encapsulates data into dataloader form
    train_loader = data.DataLoader(
        dataset=new_dataset,
        batch_size=batch_size,
        sampler=sampler#shuffle=True
    )
    return new_dataset, "", "", train_loader, test_loader, val_loader
    
def import_dataset(name, info_flags, batch_size=128, upscale=False):
    """
    Imports a given MedMNIST dataset and prints the population
    distribution for both train and test sets.
    """
    dataset = create_data_loaders(info_flags[name][4], batch_size=batch_size, upscale=upscale)
    display_set_statistics(dataset, info_flags[name], name)
    return dataset
    
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
    return info, task, n_channels, n_classes, DataClass