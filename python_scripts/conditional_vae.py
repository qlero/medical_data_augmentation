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
    def __init__(self, in_channels, n_classes, feature_size, latent_dims, img_size):
        super(ConditionalVAE, self).__init__()
        # Records the dimensional parameters
        self.channel_size = in_channels
        self.class_size = n_classes
        self.feature_size = feature_size
        self.latent_dimensions = latent_dims
        self.hidden_layer_dimensions = [32, 64, 128]
        self.fc_factor = 3
        self.image_size = img_size
        # Declares the layers of the CVAE encoder (Convolutional + FC)
        in_channels += 1
        encoder_modules = []
        for dim in self.hidden_layer_dimensions:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=dim, 
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(dim), 
                    nn.ReLU()
                )
            )
            in_channels=dim
        # Declares the hidden layers of the CVAE decoder (Convolutional + FC)
        self.decoder_input = nn.Linear(latent_dims + n_classes, 
                                       self.hidden_layer_dimensions[-1]*self.fc_factor)
        decoder_modules = []
        for layer in range(len(self.hidden_layer_dimensions),1,-1):
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
                    nn.ReLU()
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
                    self.hidden_layer_dimensions[-1],
                    self.hidden_layer_dimensions[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.BatchNorm2d(self.hidden_layer_dimensions[-1]),
                nn.ReLU(),
                nn.Conv2D(
                    self.hidden_layer_dimensions[-1],
                    out_channels=3,
                    kernel_size=3,
                    padding=1
                ),
                nn.Tanh()
            )
        )
        # Sets up the encoder and decoder
        self.encoder = nn.Sequential(*encoder_modules)
        self.decoder = nn.Sequential(*decoder_modules)
        self.fc_mu = nn.Linear(self.hidden_layer_dimensions[-1]*self.fc_factor, latent_dims)
        self.fc_var = nn.Linear(self.hidden_layer_dimensions[-1]*self.fc_factor, latent_dims)
        # Sets up the embedding input for the encoder
        self.class_embedding = nn.Linear(n_classes, img_size*img_size)
        self.data_embedding = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    def decode(self, to_decode):
        """
        Decodes an input entry (z) by casting it up the decoder network
        declared at initialization.
        """
        decoding = self.decoder_input_layer(to_decode)
        decoding = result.view(-1, self.hidden_layer_dimensions[-1], 2, 2)
        decoding = self.decoder(decoding)
        return self.decoder_output_layer(decoding)
    def encode(self, to_encode):
        """
        Encodes an input image by casting it down the encoder network
        declared at initialization.
        """
        encoding = torch.flatten(self.encoder(to_encode), start_dim=1)
        # Computes the mean and variance components of the variational autoencoder
        mu = self.fc_mu(encoding)
        var = self.fc_var(encoding)
        return mu, var
    def forward(self, x, y):
        """
        
        """
        embedded_classes = self.class_embedding(y)
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
        kld_weight = 0.005
        recons_loss = F.mse_loss(recons, x)
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
        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples
    def generate(self, x, y):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x, y)[0]


def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)

class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc1  = nn.Linear(feature_size + class_size, 512)
        self.fc21 = nn.Linear(512, latent_size)
        self.fc22 = nn.Linear(512, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, 512)
        self.fc4 = nn.Linear(512, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 28*28*3), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
    
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, n_channels=1):
    BCE = F.mse_loss(recon_x, x.view(-1, 28*28*n_channels))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch, n_classes, n_channels=1):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        labels = one_hot(labels, n_classes)
        recon_batch, mu, logvar = model(data, labels)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar, n_channels)
        loss.backward()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch, n_classes, n_channels=1):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(val_loader):
            data, labels = data.to(device), labels.to(device)
            labels = one_hot(labels, n_classes)
            recon_batch, mu, logvar = model(data, labels)
            test_loss += loss_function(recon_batch, data, mu, logvar, n_channels).detach().cpu().numpy()
            if i == 0:
                n = min(data.size(0), 5)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(-1, n_channels, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(val_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
    
    
# create a CVAE model

# hyper params
class_size = len(info_flags["pathmnist"][0]["label"])
n_channels=info_flags["pathmnist"][0]["n_channels"]
latent_size = 20
epochs = 10

train_loader = pathmnist[3]
val_loader = pathmnist[5]

model = CVAE(28*28*n_channels, latent_size, class_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(1, epochs + 1):
        train(epoch, class_size, n_channels)
        test(epoch, class_size, n_channels)
        with torch.no_grad():
            c = torch.eye(class_size, class_size).cuda()
            sample = torch.randn(class_size, latent_size).to(device)
            sample = model.decode(sample, c).cpu()
            save_image(sample.view(class_size, n_channels, 28, 28),
                       'sample_' + str(epoch) + '.png')