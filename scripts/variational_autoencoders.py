"""

This .py file contains the classes <ConditionalVAE>, <JointVAE>, <TestCVAE>, and 
<TestJointVAE>, and the functions <one_hot>, <model_evaluate>, <model_train>, 
<print_loss_convergence> and <run_encoder_pipeline> inspired in part from the 
following repositories:
    > https://github.com/AntixK/PyTorch-VAE
    > https://github.com/unnir/cVAE

Citations:
    > Subramanian, A.K, PyTorch-VAE,2020,GitHub, GitHub repository

"""

##############################
########## IMPORTS ###########
##############################

import matplotlib.pyplot as plt
import numpy as np
import os 
import torch
import torch.utils.data

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# cuda setup
global device 
device = torch.device("cuda")

##############################
########## CLASSES ###########
##############################

class ConditionalVAE(nn.Module):
    """
    Implementation of a Conditional Variational Autoencoder.
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
        self.decoder_input = nn.Linear(
            latent_dims + n_classes,
            self.hidden_layer_dimensions[-1]*self.fc_factor
        )
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
                    out_channels=self.channel_size,
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
        self.data_embedding = nn.Conv2d(
            self.channel_size, self.channel_size, kernel_size=1
        )
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
    def generate(self, x, y):
        """
        Given an input image x, returns the reconstructed image.
        """
        return self.forward(x, y)[0]
    def loss_function(self, recons, x, mu, var):
        """
        Implementation of the Conditional VAE loss function based on the
        Kullback-Leibler divergence.
        """
        kld_weight = 0.005
        recons_loss = F.mse_loss(recons, x, reduction="sum")
        kld_loss = torch.mean(-0.5 * torch.sum(
            1 + var - mu ** 2 - var.exp(), dim = 1
        ), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
    def reparametrize(self, mu, var):
        """
        Given a Gaussian latent space, provides a reparametrization factor
        """
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu + eps*std
    def sample(self, num_samples, y):
        """
        Samples from the latent space and return the corresponding
        image space map.
        """
        z = torch.randn(num_samples, self.latent_dimensions)
        z = z.to(device)
        z = torch.cat([z, y.float()], dim=1)
        samples = self.decode(z)
        return samples
    
class JointVAE(nn.Module):
    """
    Implementation of a Conditional Variational Autoencoder.
    """
    def __init__(self, in_channels, latent_dims, categorical_dims,
                 latent_min_capacity=0., latent_max_capacity=25.,
                 latent_gamma=30., latent_num_iter=25000,
                 categorical_min_capacity=0., categorical_max_capacity=25.,
                 categorical_gamma=30., categorical_num_iter=25000,
                 hidden_dims=[32, 64, 128, 256, 512],
                 temperature=0.5, anneal_rate=3e-5, anneal_interval=100, 
                 alpha=30., **kwargs):
        super(JointVAE, self).__init__()
        # Records the class parameters
        self.channel_size = in_channels
        self.latent_dimensions = latent_dims
        self.categorical_dimensions = categorical_dims
        self.temp = temperature
        self.min_temp = temperature
        self.anneal_rate = anneal_rate
        self.anneal_interval = anneal_interval
        self.alpha = alpha
        self.cont_min = latent_min_capacity
        self.cont_max = latent_max_capacity
        self.disc_min = categorical_min_capacity
        self.disc_max = categorical_max_capacity
        self.cont_gamma = latent_gamma
        self.disc_gamma = categorical_gamma
        self.cont_iter = latent_num_iter
        self.disc_iter = categorical_num_iter
        self.num_iter = 1
        self.fc_factor = 4
        if hidden_dims is None:
            self.hidden_layer_dimensions = [32, 64, 128, 256, 512]
        else:
            self.hidden_layer_dimensions = hidden_dims
        # Builds Encoder
        encoder_modules = []
        for h_dim in self.hidden_layer_dimensions:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        # Builds Decoder
        decoder_modules = []
        self.decoder_input = nn.Linear(
            self.latent_dimensions + self.categorical_dimensions,
            hidden_dims[-1] * self.fc_factor
        )
        for layer in range(len(self.hidden_layer_dimensions)-1,0,-1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_layer_dimensions[layer], 
                        self.hidden_layer_dimensions[layer - 1],
                        kernel_size=3,
                        stride = 2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(self.hidden_layer_dimensions[layer - 1]),
                    nn.LeakyReLU())
            )
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                self.hidden_layer_dimensions[0], 
                self.hidden_layer_dimensions[0],
                kernel_size = 3,
                stride = 2,
                padding = 2,
                output_padding = 0
            ),
            nn.BatchNorm2d(self.hidden_layer_dimensions[0]),
            nn.LeakyReLU(),
            nn.Conv2d(
                self.hidden_layer_dimensions[0], 
                out_channels = self.hidden_layer_dimensions[0],
                kernel_size = 3, 
                padding = 0,
                stride = 2
            ),
            nn.BatchNorm2d(self.hidden_layer_dimensions[0]),
            nn.LeakyReLU(),
            nn.Conv2d(
                self.hidden_layer_dimensions[0], 
                out_channels = self.channel_size,
                kernel_size = 3, 
                padding = 0,
                stride = 1
            ),
            nn.Tanh()
        )
        # Sets up the encoder and decoder
        self.encoder = nn.Sequential(*encoder_modules)
        self.decoder = nn.Sequential(*decoder_modules)
        self.fc_mu = nn.Linear(
            self.hidden_layer_dimensions[-1], #*self.fc_factor, 
            self.latent_dimensions
        )
        self.fc_var = nn.Linear(
            self.hidden_layer_dimensions[-1], #*self.fc_factor, 
            self.latent_dimensions
        )
        self.fc_z = nn.Linear(
            self.hidden_layer_dimensions[-1], #*self.fc_factor, 
            self.categorical_dimensions
        )
    
        self.sampling_dist = torch.distributions.OneHotCategorical(
            1./self.categorical_dimensions*torch.ones((self.categorical_dimensions, 1))
        )
    def decode(self, to_decode):
        """
        Decodes an input entry (z) by casting it up the decoder network
        declared at initialization.
        """
        decoding = self.decoder_input(to_decode)
        decoding = decoding.view(-1, self.hidden_layer_dimensions[-1], 2, 2)
        decoding = self.decoder(decoding)
        decoding = self.final_layer(decoding)
        return decoding
    def encode(self, to_encode):
        """
        Encodes an input image by casting it down the encoder network
        declared at initialization.
        """
        encoding = self.encoder(to_encode)
        encoding = torch.flatten(encoding, start_dim=1)
        # Splits the result into mu and var components of the latent Gaussian distribution
        mu = self.fc_mu(encoding)
        var = self.fc_var(encoding)
        z = self.fc_z(encoding)
        z = z.view(-1, self.categorical_dimensions)
        return mu, var, z
    def forward(self, x):
        """
        Performs a forward pass through the Joint VAE.
        """ 
        mu, var, q = self.encode(x)
        z = self.reparameterize(mu, var, q)
        return  self.decode(z), x, q, mu, var
    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image.
        """
        return self.forward(x)[0]
    def loss_function(self, recons, x, q, mu, var, batch_idx):        
        """
        Implementation of the Joint VAE loss function based on the
        Kullback-Leibler divergence.
        """
        q_p = F.softmax(q, dim=-1) # Convert the categorical codes into probabilities
        kld_weight = 0.005
        # Anneal the temperature at regular intervals
        if batch_idx % self.anneal_interval == 0 and self.training:
            self.temp = np.maximum(self.temp * np.exp(- self.anneal_rate * batch_idx),
                                   self.min_temp)
        recons_loss =F.mse_loss(recons, x, reduction='sum')
        # Adaptively increase the discrinimator capacity
        disc_curr = (self.disc_max - self.disc_min) * \
                    self.num_iter/ float(self.disc_iter) + self.disc_min
        disc_curr = min(disc_curr, np.log(self.categorical_dimensions))
        # KL divergence between gumbel-softmax distribution
        eps = 1e-7
        # Entropy of the logits
        h1 = q_p * torch.log(q_p + eps)
        # Cross entropy with the categorical distribution
        h2 = q_p * np.log(1. / self.categorical_dimensions + eps)
        kld_disc_loss = torch.mean(torch.sum(h1 - h2, dim =1), dim=0)
        # Compute Continuous loss
        # Adaptively increase the continuous capacity
        cont_curr = (self.cont_max - self.cont_min) * \
                    self.num_iter/ float(self.cont_iter) + self.cont_min
        cont_curr = min(cont_curr, self.cont_max)
        kld_cont_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(),
                                                    dim=1),
                                   dim=0)
        capacity_loss = self.disc_gamma * torch.abs(disc_curr - kld_disc_loss) + \
                        self.cont_gamma * torch.abs(cont_curr - kld_cont_loss)
        # kld_weight = 1.2
        loss = self.alpha * recons_loss + kld_weight * capacity_loss
        if self.training:
            self.num_iter += 1
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Capacity_Loss':capacity_loss}
    def reparameterize(self,mu,var,q,eps= 1e-7):
        """
        Given a Gaussian latent space, provides a reparametrization factor based 
        on the Gumbel-softmax trick to sample from Categorical Distribution.
        """
        std = torch.exp(0.5 * var)
        e = torch.randn_like(std)
        z = e * std + mu
        # Sample from Gumbel
        u = torch.rand_like(q)
        g = - torch.log(- torch.log(u + eps) + eps)
        # Gumbel-Softmax sample
        s = F.softmax((q + g) / self.temp, dim=-1)
        s = s.view(-1, self.categorical_dimensions)
        ret = torch.cat([z, s], dim=1)
        return ret
    def sample(self,num_samples, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        """
        z = torch.randn(num_samples,
                        self.latent_dimensions)
        M = num_samples
        np_y = np.zeros((M, self.categorical_dimensions), dtype=np.float32)
        np_y[range(M), np.random.choice(self.categorical_dimensions, M)] = 1
        np_y = np.reshape(np_y, [M , self.categorical_dimensions])
        q = torch.from_numpy(np_y)
        # z = self.sampling_dist.sample((num_samples * self.latent_dim, ))
        z = torch.cat([z, q], dim = 1).to(device)
        samples = self.decode(z)
        return samples
    
class TestCVAE():
    """
    Implementation of a Conditional Variational Autoencoder test.
    """
    def __init__(self, n_channels, n_classes, latent_dims):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.latent_dims = latent_dims
        self.model = ConditionalVAE(n_channels, n_classes, latent_dims)
    def test_forward(self):
        x = torch.randn(10, self.n_channels, 28, 28)
        c = torch.randn(10, self.n_classes)
        y = self.model(x, c)
        print("Model Output size:", y[0].size())
        return True
    def test_loss(self):
        x = torch.randn(10, self.n_channels, 28, 28)
        c = torch.randn(10, self.n_classes)
        result = self.model(x, c)
        loss = self.model.loss_function(*result)
        print(loss)
        return True
           
class TestJointVAE():
    """
    Implementation of a Conditional Variational Autoencoder test.
    """
    def __init__(self, n_channels, latent_dims, categorical_dims):
        self.n_channels = n_channels
        self.latent_dims = latent_dims
        self.categorical_dims = categorical_dims
        self.model = JointVAE(n_channels, latent_dims, categorical_dims)
    def test_forward(self):
        x = torch.randn(10, self.n_channels, 28, 28)
        y = self.model(x)
        print("Model Output size:", y[0].size())
        return True
    def test_loss(self):
        x = torch.randn(10, self.n_channels, 28, 28)
        decoded, x, q, mu, var = self.model(x)
        loss = self.model.loss_function(decoded, x, q, mu, var, batch_idx=5)
        print(loss)
        return True

##############################
######### FUNCTIONS ##########
##############################
 
def model_evaluate(model, loader, epoch, n_classes, n_channels, name_dataset,
                   eval_step, print_reconstruction, model_type):
    """
    Evaluates a model over a validation or test loader
    """
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(loader):
            data = data.to(device)
            if model_type == "CondVAE":
                labels = one_hot(labels.to(device), n_classes)
                recon_batch, _, mu, var = model(data, labels)
                loss = model.loss_function(recon_batch, data, mu, var)["loss"]
            else:
                recon_batch, _, q, mu, var = model(data)
                loss = model.loss_function(recon_batch, data, q, mu, var, batch_idx)["loss"]
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

def model_train(model, optimizer, train_loader, epoch, n_classes, 
                print_batch_loss, model_type):
    """
    Performs a single forward and backward pass of a given model.
    """
    length_dataset = len(train_loader.dataset)
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        if model_type == "CondVAE":
            labels = one_hot(labels.to(device), n_classes)
            recon_batch, _, mu, var = model(data, labels)
            optimizer.zero_grad()
            loss = model.loss_function(recon_batch, data, mu, var)["loss"]
        else:
            recon_batch, _, q, mu, var = model(data)
            optimizer.zero_grad()
            loss = model.loss_function(recon_batch, data, q, mu, var, batch_idx)["loss"]
        loss.backward()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()
        if batch_idx % 10 == 0 and print_batch_loss:
            print(f"Train epoch {epoch}: [{batch_idx*len(data)}/{length_dataset}]",
                  f"\tLoss: {round(loss.item()/len(data),6)}")
    print(f"Train epoch {epoch} -- average loss: {train_loss/length_dataset}")
    return model, optimizer, train_loss/length_dataset
    
def one_hot(labels, class_size):
    """
    Implements a one-hot encoding function for categorical labels.
    """
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)

def print_loss_convergence(training_losses, validation_losses, test_loss):
    """
    Prints the convergence plot of the training and validation losses.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses)
    plt.plot(validation_losses)
    plt.plot(len(training_losses)-1,test_loss,'ro')
    plt.title("Training & Validation Losses")
    plt.legend(["Training loss", "Validation loss", "Test loss (at last epoch/early stop)"])
    plt.show()

def run_encoder_pipeline(train_loader, val_loader, test_loader, 
                         n_channels, latent_dims, n_classes=2, categorical_dims=0,
                         epochs=200, name_dataset="", learning_rate=1e-3,
                         output_intermediary_info=False, model_type="CondVAE"):
    """
    Runs the training and testing process for a variational autoencoder declared above.
    """
    if model_type == "CondVAE":
        print("===================","\nTesting Conditional Variational Autoencoder:")
        model = TestCVAE(n_channels, n_classes, latent_dims)
        print("Clearing CUDA Pytorch Cache")
        torch.cuda.empty_cache()
        print("Forward pass test: ", model.test_forward())
        print("Loss test: ", model.test_loss())
        print("Cuda device: ", device)
        print("===================","\nTraining phase:")
        # Declares the model and optimizer
        model = ConditionalVAE(n_channels, n_classes, latent_dims).to(device)
    else:
        print("===================","\nTesting Joint Variational Autoencoder:")
        model = TestJointVAE(n_channels, latent_dims, categorical_dims)
        print("Clearing CUDA Pytorch Cache")
        torch.cuda.empty_cache()
        print("Forward pass test: ", model.test_forward())
        print("Loss test: ", model.test_loss())
        print("Cuda device: ", device)
        print("===================","\nTraining phase:")
        # Declares the model and optimizer
        model = JointVAE(n_channels, latent_dims, categorical_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Declares intermediary placeholders
    training_losses = []
    validation_losses = []
    validation_check = float("inf")
    validation_counter = 0
    model_to_save = None
    save_epoch = 1
    # Creates a dedicated folder for intermediary image saves to show
    # the evolution of sampling from a latent space and reconstruction
    # if indicated at function run
    if not os.path.exists(f"trained_models/{model_type}/"):
        os.makedirs(f"trained_models/{model_type}/")
    if name_dataset not in os.listdir(f"trained_models/{model_type}/"): 
        os.makedirs(f"trained_models/{model_type}/{name_dataset}")
    for epoch in range(1, epochs+1):
        # Trains
        model, optimizer, train_loss = model_train(
            model, optimizer, train_loader, epoch, n_classes,
            output_intermediary_info, model_type
        )
        # Evaluates on validation set
        model, val_loss = model_evaluate(
            model, val_loader, epoch, n_classes, n_channels, name_dataset,
            "Validation", output_intermediary_info, model_type
        )
        # Records the losses
        training_losses.append(train_loss)
        validation_losses.append(val_loss)
        # Checks if there has been improvement in the validation loss
        # to perform early stopping
        if val_loss < validation_check:
            validation_check = val_loss
            validation_counter = 0
            save_epoch = epoch
            model_to_save = model
        else:
            validation_counter += 1
        if validation_counter >= 5:
            print("==/!\== EARLY STOPPING: no validation loss",
                  "improvement over the past 5 epochs")
            break
        # Computes sampled images from the latent space if indicated
        if output_intermediary_info:
            with torch.no_grad():
                if model_type == "CondVAE":
                    c = torch.eye(n_classes, n_classes).cuda()
                    sample = torch.randn(n_classes, latent_dims).to(device)
                    sample = torch.cat([sample, c], dim=1)
                    sample = model.decode(sample).cpu()
                    save_image(sample.view(n_classes, n_channels, 28, 28),
                               f"trained_models/{model_type}/{name_dataset}/epoch{epoch}_sample.png")
                else:
                    sample = model.sample(10)
                    save_image(sample.view(10, n_channels, 28, 28),
                               f"trained_models/{model_type}/{name_dataset}/epoch{epoch}_sample.png")
    print("===================","\nTesting phase:")
    if model_to_save is None:
        model_to_save = model
        save_epoch = 200
    model, test_loss = model_evaluate(
        model_to_save, test_loader, epoch, n_classes, n_channels, name_dataset,
        "Test", output_intermediary_info, model_type
    )
    # Prints the training and validation convergence
    print_loss_convergence(training_losses, validation_losses, test_loss)
    # Saves the model
    torch.save(model.state_dict(),
               f"trained_models/{model_type}/{name_dataset}/{name_dataset}_epoch{save_epoch}.pth")
    return model, training_losses, validation_losses, test_loss
    
    
    