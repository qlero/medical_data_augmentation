"""

This .py file contains the classes 
inspired in part from the following repositories:
    > https://github.com/imadtoubal/Conditional-DC-GAN-in-Pytorch

"""

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.utils

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Resize
from tqdm import tqdm

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, bn=False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_features, 
                out_features, 
                3, 1, 0
            ),
            nn.BatchNorm2d(out_features) if bn else nn.Identity(),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.layers(x)

class TransConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride = 1, padding = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_features, 
                out_features, 
                4,
                stride=stride, 
                padding=padding
            ),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.layers(x)

class Discriminator(nn.Module):
    def __init__(self, n_channels, ndf = 28, img_size = 28):
        super().__init__()
        self.img_size = img_size
        self.layers = nn.Sequential(
            ConvBlock(
                n_channels, 
                ndf, 
                bn=False
            ),
            ConvBlock(
                ndf, 
                ndf * 2
            ),
            ConvBlock(
                ndf * 2, 
                ndf * 4
            ),
            nn.Conv2d(
                ndf * 4, 
                1, 
                3, 
                2, 
                0, 
            ),
            nn.MaxPool2d(3,2,1),
            nn.Conv2d(
                1,1,5
            ),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)
    
# class Discriminator(nn.Module):
#     def __init__(self, n_channels, ndf = 28, img_size = 28):
#         super().__init__()
#         self.img_size = img_size
#         self.layers = nn.Sequential(
#             ConvBlock(
#                 n_channels, 
#                 ndf, 
#                 bn=False
#             ),
#             ConvBlock(
#                 ndf, 
#                 ndf *2
#             ),
#             ConvBlock(
#                 ndf * 2,
#                 ndf * 4
#             ),
#             ConvBlock(
#                 ndf * 4, 
#                 ndf * 8
#             ),
#             nn.Conv2d(
#                 ndf * 8, 
#                 1, 
#                 5, 
#                 3, 
#                 0, 
#                 bias=False
#             ),
#             nn.Conv2d(
#                 1, 
#                 1, 
#                 3, 
#                 3, 
#                 0, 
#                 bias=False
#             ),
#             nn.MaxPool2d(3, 2, 1),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         return self.layers(x)

class Generator(nn.Module):
    def __init__(self, z_dim, n_channels, ngf = 28, img_size = 28):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.layers = nn.Sequential(
            TransConvBlock(
                z_dim, 
                ngf * 8, 
                stride=2, 
                padding=1
            ),
            TransConvBlock(
                ngf * 8, 
                ngf * 4,
                stride=2,
                padding=1
            ),
            TransConvBlock(
                ngf * 4, 
                ngf * 4,
                stride=2,
                padding=0
            ),
            nn.ConvTranspose2d(
                ngf*4, 
                n_channels, 
                3, 
                3, 
                1,
            ),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

# class Generator(nn.Module):
#     def __init__(self, z_dim, n_channels, ngf = 28, img_size = 28):
#         super(Generator, self).__init__()
#         self.img_size = img_size
#         self.layers = nn.Sequential(
#             TransConvBlock(
#                 z_dim, 
#                 ngf * 8, 
#                 stride=2, 
#                 padding=0
#             ),  # 4 x 4
#             TransConvBlock(
#                 ngf * 8, 
#                 ngf * 4
#             ),  # 8 x 8
#             TransConvBlock(
#                 ngf * 4, 
#                 ngf * 4,
#                 stride=2
#             ),  # 8 x 8
#             TransConvBlock(
#                 ngf * 4, 
#                 ngf * 2
#             ),  # 16 x 16
#             TransConvBlock(
#                 ngf * 2, 
#                 ngf * 2
#             ),
#             TransConvBlock(
#                 ngf * 2, 
#                 ngf * 2
#             ),
#             TransConvBlock(
#                 ngf * 2, 
#                 ngf * 1
#             ),  # 32 x 32
#             nn.ConvTranspose2d(
#                 ngf, n_channels, 
#                 4, 
#                 2, 
#                 1,
#             ),  # 64 x 64
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         return self.layers(x)

class ConditionalDiscriminator(Discriminator):
    def __init__(self, n_channels, n_classes, ndf = 28, img_size = 28):
        super().__init__(n_channels + 1, ndf=ndf, img_size=img_size)
        self.embed = nn.Embedding(n_classes, self.img_size ** 2)
    def forward(self, x, labels):
        x = torch.cat(
            [x,self.embed(labels).view(-1, 1, self.img_size, self.img_size)], 
            dim=1
        )
        return super().forward(x)

class ConditionalGenerator(Generator):
    def __init__(self, z_dim, n_channels, n_classes, ngf = 32) -> None:
        super().__init__(z_dim * 2, n_channels, ngf=ngf)
        self.embed = nn.Embedding(n_classes, z_dim)
    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2)#.unsqueeze(3)
        cat = torch.cat(
            [x, embedding.permute(0, 3, 1, 2)], 
            dim=1
        )
        return super().forward(cat)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Block') == -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def model_evaluate(model, loader, epoch, n_classes, n_channels, name_dataset,
                   eval_step, print_reconstruction):
    """
    Evaluates a model over a validation or test loader
    """
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(loader):
            data = data.to(device)
            labels = one_hot(labels.to(device), n_classes)
            recon_batch, _, mu, var = model(data, labels)
            loss = model.loss_function(recon_batch, data, mu, var)["loss"]
            eval_loss += loss.detach().cpu().numpy()
            if batch_idx == 0 and print_reconstruction:
                n = min(data.size(0), 5)
                comparison = torch.cat(
                    [data[:n], recon_batch.view(-1, n_channels, 28, 28)[:n]]
                )
                save_image(comparison.cpu(), 
                           f"trained_models/{model_type}/{name_dataset}/epoch{epoch}_recons.png",
                           nrow=n)
    eval_loss /= len(loader.dataset)
    print(f"{eval_step} set loss: {round(eval_loss,6)}")
    return model, eval_loss

def model_train(device, generator, discriminator, 
                optimizer_generator, optimizer_discriminator,
                train_loader,
                batch_size, latent_dims, n_classes, epochs, print_batch_loss):
    """
    Performs a single forward and backward pass of a given model.
    """
    length_dataset = len(train_loader.dataset)
    bce_loss = nn.BCELoss()
    train_generator_loss = 0
    train_discriminator_loss = 0       
    fixed_noise = torch.randn(batch_size * latent_dims)
    fixed_noise = fixed_noise.to(device).view(batch_size, latent_dims, 1, 1)
    for epoch in range(epochs):
        for batch_idx, (real, labels) in enumerate(train_loader):
            real = real.to(device)
            labels = labels.to(device)
            batch_size = real.shape[0]
            # Train descriminator: minimize the classification loss
            noise = torch.randn(batch_size * latent_dims)
            noise = noise.view(batch_size, latent_dims, 1, 1).to(device)
            fake = generator(noise, labels)
            disc_real = discriminator(real, labels).view(-1)
            disc_fake = discriminator(fake, labels).view(-1)
            loss_real = bce_loss(disc_real, torch.ones_like(disc_real))
            loss_fake = bce_loss(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = (loss_real + loss_fake) / 2
            discriminator.zero_grad()
            disc_loss.backward(retain_graph=True)
            optimizer_discriminator.step()
            # Train generator: maximize the classification loss for fake images
            output = discriminator(fake, labels)
            gen_loss = bce_loss(output, torch.ones_like(output))
            generator.zero_grad()
            gen_loss.backward()
            optimizer_generator.step()
            train_generator_loss += gen_loss
            train_discriminator_loss += disc_loss
            if batch_idx % 10 == 0 and print_batch_loss:
                print(f"Train epoch {epoch}: [{batch_idx*len(data)}/{length_dataset}]",
                      f"\tLoss: {round(loss.item()/len(data),6)}")
    avg_gen_loss = train_generator_loss/length_dataset
    avg_dis_loss = train_discriminator_loss/length_dataset
    print(f"Train epoch {epoch}:",
          f"avg. discriminator loss: {avg_dis_loss}, ",
          f"avg. generator loss: {avg_gen_loss}")
    return generator, discriminator, avg_gen_loss, avg_dis_loss
        
def run_GAN_pipeline(train_loader, val_loader, test_loader, 
                     n_channels, latent_dims, n_classes, batch_size,
                     epochs=200, name_dataset="", learning_rate=1e-4,
                     output_intermediary_info=False):
    # Hyper parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Declares models
    discriminator = ConditionalDiscriminator(
        n_channels=n_channels, 
        n_classes=n_classes
    ).to(device)
    generator = ConditionalGenerator(
        z_dim=latent_dims, 
        n_channels=n_channels, 
        n_classes=n_classes
    ).to(device)

    # Initializes weights
    discriminator.apply(weights_init)
    generator.apply(weights_init)
    
    # Initializes the optimizers
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    opt_gen = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    
    # Trains
    generator, discriminator, avg_gen_loss, avg_dis_loss = model_train(
        device, generator, discriminator, opt_gen, opt_disc,
        train_loader,
        batch_size, latent_dims, n_classes, epochs, 
        print_batch_loss=output_intermediary_info
    )