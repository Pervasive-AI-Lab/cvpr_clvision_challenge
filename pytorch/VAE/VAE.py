from __future__ import print_function
import pdb
import numpy as np

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import Variable
import VAE.flows as flows
from VAE.layers import GatedConv2d, GatedConvTranspose2d
from utils import Reshape

class VAE(nn.Module):
    """
    The base VAE class containing gated convolutional encoder and decoder architecture.
    Can be used as a base class for VAE's with normalizing flows.
    """

    def __init__(self, args):
        super(VAE, self).__init__()

        # extract model settings from args
        self.args = args
        self.z_size = args.z_size
        self.input_size = args.input_size
        self.input_type = args.input_type
        self.gen_hiddens = args.gen_hiddens


        if self.input_size == [1, 28, 28] or self.input_size == [3, 28, 28]:
            self.last_kernel_size = 7
        elif self.input_size == [1, 28, 20]:
            self.last_kernel_size = (7, 5)
        elif self.input_size == [3, 32, 32]:
            self.last_kernel_size = 8
        else:
            if self.args.dataset=='permuted_mnist':
                # this dataset has no 3D structure
                assert self.input_size == [784]
                assert self.args.gen_architecture == 'MLP'
            else:
                raise ValueError('invalid input size!!')

        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        self.p_x_nn, self.p_x_mean = self.create_decoder()

        self.q_z_nn_output_dim = 256

        # auxiliary
        if args.cuda:
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

        # log-det-jacobian = 0 without flows
        self.log_det_j = Variable(self.FloatTensor(1).zero_())

        self.prior = MultivariateNormal(torch.zeros(args.z_size), torch.eye(args.z_size))

        # get gradient dimension:
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())

    def create_encoder(self):
        """
        Helper function to create the elemental blocks for the encoder. Creates a gated convnet encoder.
        the encoder expects data as input of shape (batch_size, num_channels, width, height).
        """

        if self.input_type == 'binary':

            if self.args.gen_architecture == 'GatedConv':
                q_z_nn = nn.Sequential(
                    GatedConv2d(self.input_size[0], 32, 5, 1, 2),
                    nn.Dropout(self.args.dropout),
                    GatedConv2d(32, 32, 5, 2, 2),
                    nn.Dropout(self.args.dropout),
                    GatedConv2d(32, 64, 5, 1, 2),
                    nn.Dropout(self.args.dropout),
                    GatedConv2d(64, 64, 5, 2, 2),
                    nn.Dropout(self.args.dropout),
                    GatedConv2d(64, 64, 5, 1, 2),
                    nn.Dropout(self.args.dropout),
                    GatedConv2d(64, self.gen_hiddens, self.last_kernel_size, 1, 0),
                )
                assert self.args.gen_depth == 6

            elif self.args.gen_architecture == 'MLP':
                q_z_nn = [
                    Reshape([-1]),
                    nn.Linear(np.prod(self.args.input_size), self.gen_hiddens),
                    nn.ReLU(True),
                    nn.Dropout(self.args.dropout),
                ]
                for i in range(1, self.args.gen_depth):
                    q_z_nn += [
                        nn.Linear(self.args.gen_hiddens, self.args.gen_hiddens),
                        nn.ReLU(True),
                        nn.Dropout(self.args.dropout),
                    ]
                q_z_nn = nn.Sequential(*q_z_nn)

            q_z_mean = nn.Linear(self.gen_hiddens, self.z_size)
            q_z_var = nn.Sequential(
                nn.Linear(self.gen_hiddens, self.z_size),
                nn.Softplus(),
            )
            return q_z_nn, q_z_mean, q_z_var

        #TODO(add log_logistic loss for continuous)
        elif self.input_type in ['multinomial', 'continuous']:
            act = None

            q_z_nn = nn.Sequential(
                GatedConv2d(self.input_size[0], 32, 5, 1, 2, activation=act),
                GatedConv2d(32, 32, 5, 2, 2, activation=act),
                GatedConv2d(32, 64, 5, 1, 2, activation=act),
                GatedConv2d(64, 64, 5, 2, 2, activation=act),
                GatedConv2d(64, 64, 5, 1, 2, activation=act),
                GatedConv2d(64, 256, self.last_kernel_size, 1, 0, activation=act)
            )
            q_z_mean = nn.Linear(256, self.z_size)
            q_z_var = nn.Sequential(
                nn.Linear(256, self.z_size),
                nn.Softplus(),
                nn.Hardtanh(min_val=0.01, max_val=7.)

            )
            return q_z_nn, q_z_mean, q_z_var

    def create_decoder(self):
        """
        Helper function to create the elemental blocks for the decoder. Creates a gated convnet decoder.
        """

        # TODO(why the hell would num_classes be 256?)
        #num_classes = 256
        num_classes = 1

        if self.input_type == 'binary':
            if self.args.gen_architecture == 'GatedConv':
                p_x_nn = nn.Sequential(
                    Reshape([self.args.z_size, 1, 1]),
                    GatedConvTranspose2d(self.z_size, 64, self.last_kernel_size, 1, 0),
                    GatedConvTranspose2d(64, 64, 5, 1, 2),
                    GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
                    GatedConvTranspose2d(32, 32, 5, 1, 2),
                    GatedConvTranspose2d(32, 32, 5, 2, 2, 1),
                    GatedConvTranspose2d(32, 32, 5, 1, 2)
                )
                p_x_mean = nn.Sequential(
                    nn.Conv2d(32, self.input_size[0], 1, 1, 0),
                    nn.Sigmoid()
                )
            elif self.args.gen_architecture=='MLP':
                p_x_nn = [
                    nn.Linear(self.z_size, self.gen_hiddens),
                    nn.ReLU(True),
                    nn.Dropout(self.args.dropout),
                ]
                for i in range(1, self.args.gen_depth):
                    p_x_nn += [
                        nn.Linear(self.args.gen_hiddens, self.args.gen_hiddens),
                        nn.ReLU(True),
                        nn.Dropout(self.args.dropout),
                    ]
                p_x_nn = nn.Sequential(*p_x_nn)

                p_x_mean = nn.Sequential(
                    nn.Linear(self.args.gen_hiddens, np.prod(self.args.input_size)),
                    nn.Sigmoid(),
                    Reshape(self.args.input_size)
                )
            return p_x_nn, p_x_mean

        #TODO(add log_logistic loss for continuous)
        elif self.input_type in ['multinomial', 'continuous']:
            act = None
            p_x_nn = nn.Sequential(
                GatedConvTranspose2d(self.z_size, 64, self.last_kernel_size, 1, 0, activation=act),
                GatedConvTranspose2d(64, 64, 5, 1, 2, activation=act),
                GatedConvTranspose2d(64, 32, 5, 2, 2, 1, activation=act),
                GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1, activation=act),
                GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act)
            )

            p_x_mean = nn.Sequential(
                nn.Conv2d(32, 256, 5, 1, 2),
                nn.Conv2d(256, self.input_size[0] * num_classes, 1, 1, 0),
                # output shape: batch_size, num_channels * num_classes, pixel_width, pixel_height
            )

            return p_x_nn, p_x_mean

        else:
            raise ValueError('invalid input type!!')

    def reparameterize(self, mu, var):
        """
        Samples z from a multivariate Gaussian with diagonal covariance matrix using the
         reparameterization trick.
        """

        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        z = eps.mul(std).add_(mu)

        return z

    def encode(self, x):
        """
        Encoder expects following data shapes as input: shape = (batch_size, num_channels, width, height)
        """

        h = self.q_z_nn(x)
        h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)

        return mean, var

    def decode(self, z):
        """
        Decoder outputs reconstructed image in the following shapes:
        x_mean.shape = (batch_size, num_channels, width, height)
        """

        h = self.p_x_nn(z)
        x_mean = self.p_x_mean(h)

        return x_mean

    def generate(self, N=16):
        """
        Decoder outputs reconstructed image in the following shapes:
        x_mean.shape = (batch_size, num_channels, width, height)
        """

        z = self.prior.sample((N,))
        if self.args.cuda: z = z.to(self.args.device)
        h = self.p_x_nn(z)
        x_mean = self.p_x_mean(h)

        return x_mean

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log det jacobian is zero
         for a plain VAE (without flows), and z_0 = z_k.
        """

        # mean and variance of z
        z_mu, z_var = self.encode(x)
        # sample z
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z)

        return x_mean, z_mu, z_var, self.log_det_j, z, z

