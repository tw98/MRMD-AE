import torch.nn as nn
import numpy as np


class Encoder_basic(nn.Module):
    # Basic MLP encoder block
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Encoder_basic,self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.leaky_relu = nn.LeakyReLU(0.1)
        # self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        hidden1 = self.leaky_relu(self.linear(x))
        hidden2 = self.leaky_relu(self.linear2(hidden1))
        hidden3 = self.leaky_relu(self.linear3(hidden2))

        return hidden3


class Decoder_basic(nn.Module):
    # Basic MLP decoder block
    def __init__(self, hidden_dim3, hidden_dim2, hidden_dim1, output_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim3, hidden_dim2)
        self.linear2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.linear3 = nn.Linear(hidden_dim1, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)
    def forward(self,x):
        hidden2 = self.leaky_relu(self.linear(x))
        hidden1 = self.leaky_relu(self.linear2(hidden2))
        output = self.leaky_relu(self.linear3(hidden1))
        return output


class Decoder_Manifold(nn.Module):
    # Decoder with individual bottleneck manifold embedding layer
    def __init__(self, manifold_dim, hidden_dim3, hidden_dim2, hidden_dim1, output_dim):
        """
        :param manifold_dim: the bottleneck manifold dimensions, e.g. phate/tphate dim
        :param hidden_dim3: the output dim from encoder, this is reused to get symmetric encoder/decoder
        :param hidden_dim2: symmetric to encoder hidden_dim2
        :param hidden_dim1: symmetric to encoder hidden_dim1
        :param output_dim: the reconstructed dim same as encoder input_dim
        """
        super(Decoder_Manifold, self).__init__()
        self.linear_m = nn.Linear(hidden_dim3, manifold_dim)
        self.linear = nn.Linear(manifold_dim, hidden_dim3 )
        self.linear_1 = nn.Linear(hidden_dim3, hidden_dim2)
        self.linear_2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.linear_3 = nn.Linear(hidden_dim1, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)
    def forward(self, x):
        z = self.linear_m(x)
        hidden3 = self.leaky_relu(self.linear(z))
        hidden2 = self.leaky_relu(self.linear_1(hidden3))
        hidden1 = self.leaky_relu(self.linear_2(hidden2))
        output = self.leaky_relu(self.linear_3(hidden1))
        return z, output


class Encoder_basic_btlnk(nn.Module):
    # add a bottleneck layer to the encoder, so that the encoder output the same bottleneck dim as the manifold
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, manifold_dim):
        super(Encoder_basic_btlnk,self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.linear_m = nn.Linear(hidden_dim3, manifold_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)
        # self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        hidden1 = self.leaky_relu(self.linear(x))
        hidden2 = self.leaky_relu(self.linear2(hidden1))
        hidden3 = self.leaky_relu(self.linear3(hidden2))
        hidden_m = self.leaky_relu(self.linear_m(hidden3))

        return hidden_m


class Decoder_Manifold_btlnk(nn.Module):
    # Decoder with individual bottleneck manifold embedding layer
    def __init__(self, manifold_dim, hidden_dim3, hidden_dim2, hidden_dim1, output_dim):
        """
        :param manifold_dim: the bottleneck manifold dimensions, e.g. phate/tphate dim
        :param hidden_dim3: symmetric to encoder hidden_dim3
        :param hidden_dim2: symmetric to encoder hidden_dim2
        :param hidden_dim1: symmetric to encoder hidden_dim1
        :param output_dim: the reconstructed dim same as encoder input_dim
        """
        super(Decoder_Manifold_btlnk, self).__init__()
        self.linear_m = nn.Linear(manifold_dim, manifold_dim) #the input to this is the same bottleneck dimension of manifold dim
        self.linear = nn.Linear(manifold_dim, hidden_dim3)
        self.linear_1 = nn.Linear(hidden_dim3, hidden_dim2)
        self.linear_2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.linear_3 = nn.Linear(hidden_dim1, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)
    def forward(self, x):
        z = self.linear_m(x)
        hidden3 = self.leaky_relu(self.linear(z))
        hidden2 = self.leaky_relu(self.linear_1(hidden3))
        hidden1 = self.leaky_relu(self.linear_2(hidden2))
        output = self.leaky_relu(self.linear_3(hidden1))
        return z, output