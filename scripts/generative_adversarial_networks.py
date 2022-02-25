"""

This .py file contains the function <run_GAN_pipeline> 
inspired in part from the following repository:
    > https://github.com/imadtoubal/Conditional-DC-GAN-in-Pytorch

"""

import os
import torch.cuda as cuda
import torch.nn as nn
from torchfusion.gan.learners import *
from torchfusion.gan.applications import StandardGenerator, StandardProjectionDiscriminator
from torch.optim import Adam

##############################
######### FUNCTIONS ##########
##############################

def run_GAN_pipeline(
    train_loader, val_loader, test_loader, 
    n_channels, latent_dims, n_classes,
    epochs=200, name_dataset="", learning_rate=1e-4
):
    # Defines generator and discriminator
    G = StandardGenerator(
        output_size=(n_channels,32,32),
        latent_size=latent_dims,
        num_classes=n_classes
    )
    D = StandardProjectionDiscriminator(
        input_size=(n_channels,32,32),
        apply_sigmoid=False,
        num_classes=n_classes
    )
    # Moves to GPU if available
    if cuda.is_available():
        G = nn.DataParallel(G.cuda())
        D = nn.DataParallel(D.cuda())
    # Setups optimizers
    g_optim = Adam(
        G.parameters(),
        lr=learning_rate,
        betas=(0.5,0.999)
    )
    d_optim = Adam(
        D.parameters(),
        lr=learning_rate,
        betas=(0.5,0.999)
    )
    # Initializes the learner
    learner = RAvgStandardGanLearner(G,D)
    # Creates path
    path = f"trained_models/conditionalGAN/{name_dataset}/"
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Path {path} created.")
    # Trains
    learner.train(
        train_loader,
        gen_optimizer=g_optim,
        disc_optimizer=d_optim,
        num_classes=n_classes,
        save_outputs_interval=500,
        model_dir=path,
        latent_size=latent_dims,
        num_epochs=epochs,
        batch_log=False
    )