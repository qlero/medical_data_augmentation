"""



"""

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# cuda setup
device = torch.device("cuda")

class ConditionalVAE(nn.Module):
    """
    Implementation of a Conditional Variational Autoencoder inspired from
    the following implementations:
    > https://github.com/unnir/cVAE
    > https://github.com/AntixK/PyTorch-VAE
    """
    def __init__(self, in_channels, n_classes, latent_dims, 
                 hidden_dims=[32, 64, 128, 256, 512], img_size=28):
        super(ConditionalVAE, self).__init__()
        # Records the dimensional parameters
        self.channel_size = in_channels
        self.class_size = n_classes
        self.latent_dimensions = latent_dims
        self.hidden_layer_dimensions = hidden_dims
        self.fc_factor = 4
        self.image_size = img_size
        # Declares the layers of the CVAE encoder (Convolutional + FC)
        in_channels += 1
        encoder_modules = []
        for dim in self.hidden_layer_dimensions:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=dim, 
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(dim), 
                    nn.LeakyReLU()
                )
            )
            in_channels=dim
        # Declares the hidden layers of the CVAE decoder (Convolutional + FC)
        self.decoder_input = nn.Linear(latent_dims + n_classes, 
                                       self.hidden_layer_dimensions[-1]*self.fc_factor)
        decoder_modules = []
        for layer in range(len(self.hidden_layer_dimensions)-1,0,-1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_layer_dimensions[layer],
                        self.hidden_layer_dimensions[layer-1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(self.hidden_layer_dimensions[layer-1]),
                    nn.LeakyReLU()
                )
            )
        # Declares the first layer of the decoder 
        self.decoder_input_layer = nn.Linear(
            latent_dims + n_classes,
            self.hidden_layer_dimensions[-1]*self.fc_factor
        )
        # Declares the final layer of the decoder
        self.decoder_output_layer = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.hidden_layer_dimensions[0],
                    self.hidden_layer_dimensions[0],
                    kernel_size=3,
                    stride=2,
                    padding=2,
                    output_padding=0
                ),
                nn.BatchNorm2d(self.hidden_layer_dimensions[0]),
                nn.LeakyReLU(),
                nn.Conv2d(
                    self.hidden_layer_dimensions[0],
                    out_channels=self.hidden_layer_dimensions[0],
                    kernel_size=3,
                    padding=0,
                    stride=2,
                ),
                nn.BatchNorm2d(self.hidden_layer_dimensions[0]),
                nn.LeakyReLU(),
                nn.Conv2d(
                    self.hidden_layer_dimensions[0],
                    out_channels=3,
                    kernel_size=3,
                    padding=0,
                    stride=1
                ),
                nn.Tanh()
            )
        )
        # Sets up the encoder and decoder
        self.encoder = nn.Sequential(*encoder_modules)
        self.decoder = nn.Sequential(*decoder_modules)
        self.fc_mu = nn.Linear(self.hidden_layer_dimensions[-1], latent_dims)
        self.fc_var = nn.Linear(self.hidden_layer_dimensions[-1], latent_dims)
        # Sets up the embedding input for the encoder
        self.class_embedding = nn.Linear(n_classes, img_size*img_size)
        self.data_embedding = nn.Conv2d(self.channel_size, self.channel_size, kernel_size=1)
    def decode(self, to_decode):
        """
        Decodes an input entry (z) by casting it up the decoder network
        declared at initialization.
        """
        decoding = self.decoder_input_layer(to_decode)
        decoding = decoding.view(-1, self.hidden_layer_dimensions[-1], 2, 2)
        decoding = self.decoder(decoding)
        decoding = self.decoder_output_layer(decoding)
        return decoding
    def encode(self, to_encode):
        """
        Encodes an input image by casting it down the encoder network
        declared at initialization.
        """
        encoding = self.encoder(to_encode)
        encoding = torch.flatten(encoding, start_dim=1)
        # Computes the mean and variance components of the variational autoencoder
        mu = self.fc_mu(encoding)
        var = self.fc_var(encoding)
        return mu, var
    def forward(self, x, y):
        """
        Performs a forward pass through the Conditional VAE
        """
        embedded_classes = self.class_embedding(y.float())
        embedded_classes = embedded_classes.view(
            -1,self.image_size,self.image_size
        ).unsqueeze(1) 
        embedded_input = self.data_embedding(x)
        _x = torch.cat([embedded_input, embedded_classes], dim=1)
        mu, var = self.encode(_x)
        z = self.reparametrize(mu, var)
        z = torch.cat([z, y], dim=1)
        return self.decode(z), x, mu, var
    def reparametrize(self, mu, var):
        """
        Given a Gaussian latent space, provides a reparametrization factor
        """
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu + eps*std
    def loss_function(self, recons, x, mu, var):
        """
        Implementation of the Conditional VAE loss function based on the
        Kullback-Leibler divergence.
        """
        kld_weight = 0.005
        recons_loss = F.mse_loss(recons, x, reduction="sum")
        kld_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
    def sample(self, num_samples, current_device, y):
        """
        Samples from the latent space and return the corresponding
        image space map.
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        z = torch.cat([z, y.float()], dim=1)
        samples = self.decode(z)
        return samples
    def generate(self, x, y):
        """
        Given an input image x, returns the reconstructed image.
        """
        return self.forward(x, y)[0]


class TestCVAE():
    """
    Implementation of a Conditional Variational Autoencoder test inspired from
    the following implementations:
    > https://github.com/unnir/cVAE
    > https://github.com/AntixK/PyTorch-VAE
    """
    def setUp(self) -> None:
        self.model = ConditionalVAE(3, 10, 10)
    def test_forward(self):
        x = torch.randn(2, 3, 28, 28)
        c = torch.randn(2, 10)
        y = self.model(x, c)
        print("Model Output size:", y[0].size())
    def test_loss(self):
        x = torch.randn(2, 3, 28, 28)
        c = torch.randn(2, 10)
        result = self.model(x, c)
        loss = self.model.loss_function(*result)
        print(loss)   