import os
import sys
from functools import partial

import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torchinfo import summary
from libs.unet import Identity
from libs.utils import *

current_path = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(current_path)
sys.path.append(SRC_ROOT)

class SpectralConv1d(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 modes: int,
                 n_grid=None,
                 dropout=0.1,
                 return_freq=False,
                 activation='silu',
                 debug=False):
        super(SpectralConv1d, self).__init__()

        '''
        Modified Zongyi Li's Spectral1dConv code
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py
        '''

        self.linear = nn.Linear(in_dim, out_dim)  # for residual
        self.modes = modes
        activation = default(activation, 'silu')
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        self.n_grid = n_grid  
        self.fourier_weight = Parameter(
            torch.FloatTensor(in_dim, out_dim, modes, 2))
        xavier_normal_(self.fourier_weight, gain=1/(in_dim*out_dim))
        self.dropout = nn.Dropout(dropout)
        self.return_freq = return_freq
        self.debug = debug

    @staticmethod
    def complex_matmul_1d(a, b):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        op = partial(torch.einsum, "bix,iox->box")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        '''
        Input: (-1, n_grid, in_features)
        Output: (-1, n_grid, out_features)
        '''
        seq_len = x.size(1)
        res = self.linear(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        x_ft = fft.rfft(x, n=seq_len, norm="ortho")
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        out_ft = self.complex_matmul_1d(
            x_ft[:, :, :self.modes], self.fourier_weight)

        pad_size = seq_len//2 + 1 - self.modes
        out_ft = F.pad(out_ft, (0, 0, 0, pad_size), "constant", 0)

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = fft.irfft(out_ft, n=seq_len, norm="ortho")

        x = x.permute(0, 2, 1)
        x = self.activation(x + res)

        if self.return_freq:
            return x, out_ft
        else:
            return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 modes: int,  # number of fourier modes
                 n_grid=None,
                 dropout=0.1,
                 norm='ortho',
                 activation='silu',
                 return_freq=False,  # whether to return the frequency target
                 debug=False):
        super(SpectralConv2d, self).__init__()

        '''
        Modified Zongyi Li's SpectralConv2d PyTorch 1.6 code
        using only real weights
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        '''
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)  # for residual
        self.modes = modes
        activation = default(activation, 'silu')
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        self.n_grid = n_grid  # just for debugging
        self.fourier_weight = nn.ParameterList([Parameter(
            torch.FloatTensor(in_dim, out_dim,
                                                modes, modes, 2)) for _ in range(2)])
        for param in self.fourier_weight:
            xavier_normal_(param, gain=1/(in_dim*out_dim)
                           * np.sqrt(in_dim+out_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = norm
        self.return_freq = return_freq
        self.debug = debug

    @staticmethod
    def complex_matmul_2d(a, b):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        '''
        Input: (-1, n_grid**2, in_features) or (-1, n_grid, n_grid, in_features)
        Output: (-1, n_grid**2, out_features) or (-1, n_grid, n_grid, out_features)
        '''
        batch_size = x.size(0)
        n_dim = x.ndim
        if n_dim == 4:
            n = x.size(1)
            assert x.size(1) == x.size(2)
        elif n_dim == 3:
            n = int(x.size(1)**(0.5))
        else:
            raise ValueError("Dimension not implemented")
        in_dim = self.in_dim
        out_dim = self.out_dim
        modes = self.modes

        x = x.view(-1, n, n, in_dim)
        res = self.linear(x)
        x = self.dropout(x)

        x = x.permute(0, 3, 1, 2)
        x_ft = fft.rfft2(x, s=(n, n), norm=self.norm)
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        out_ft = torch.zeros(batch_size, out_dim, n, n //
                             2+1, 2, device=x.device)
        out_ft[:, :, :modes, :modes] = self.complex_matmul_2d(
            x_ft[:, :, :modes, :modes], self.fourier_weight[0])
        out_ft[:, :, -modes:, :modes] = self.complex_matmul_2d(
            x_ft[:, :, -modes:, :modes], self.fourier_weight[1])
        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = fft.irfft2(out_ft, s=(n, n), norm=self.norm)
        x = x.permute(0, 2, 3, 1)
        x = self.activation(x + res)

        if n_dim == 3:
            x = x.view(batch_size, n**2, out_dim)

        if self.return_freq:
            return x, out_ft
        else:
            return x

class FourierNeuralOperator(nn.Module):
    def __init__(self, in_dim,
                 n_hidden,
                 freq_dim,
                 out_dim,
                 modes: int,
                 num_spectral_layers: int = 2,
                 n_grid=None,
                 dim_feedforward=None,
                 spacial_fc=True,
                 spacial_dim=2,
                 return_freq=False,
                 return_latent=False,
                 normalizer=None,
                 activation='silu',
                 last_activation=True,
                 add_grad_channel=False,
                 dropout=0.1,
                 debug=False,
                 **kwargs):
        super(FourierNeuralOperator, self).__init__()
        '''
        A wrapper for both SpectralConv1d and SpectralConv2d
        Ref: Li et 2020 FNO paper
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        A new implementation incoporating all spacial-based FNO
        in_dim: input dimension, (either n_hidden or spacial dim)
        n_hidden: number of hidden features out from spatial to the fourier conv
        freq_dim: frequency feature dim
        '''
        if spacial_dim == 2:  # 2d, function + (x,y)
            spectral_conv = SpectralConv2d
        elif spacial_dim == 1:  # 1d, function + x
            spectral_conv = SpectralConv1d
        else:
            raise NotImplementedError("3D not implemented.")
        activation = default(activation, 'silu')
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        dropout = default(dropout, 0.1)
        self.spacial_fc = spacial_fc
        if self.spacial_fc:
            self.fc = nn.Linear(in_dim + spacial_dim, n_hidden)
        self.spectral_conv = nn.ModuleList([spectral_conv(in_dim=n_hidden,
                                                          out_dim=freq_dim,
                                                          n_grid=n_grid,
                                                          modes=modes,
                                                          dropout=dropout,
                                                          activation=activation,
                                                          return_freq=return_freq,
                                                          debug=debug)])
        for _ in range(num_spectral_layers - 1):
            self.spectral_conv.append(spectral_conv(in_dim=freq_dim,
                                                    out_dim=freq_dim,
                                                    n_grid=n_grid,
                                                    modes=modes,
                                                    dropout=dropout,
                                                    activation=activation,
                                                    return_freq=return_freq,
                                                    debug=debug))
        if not last_activation:
            self.spectral_conv[-1].activation = Identity()

        self.n_grid = n_grid  # dummy for debug
        self.dim_feedforward = default(dim_feedforward, 2*spacial_dim*freq_dim)
        self.regressor = nn.Sequential(
            nn.Linear(freq_dim, self.dim_feedforward),
            self.activation,
            nn.Linear(self.dim_feedforward, out_dim),
        )
        self.normalizer = normalizer
        self.return_freq = return_freq
        self.return_latent = return_latent
        self.add_grad_channel = add_grad_channel
        self.debug = debug

    def forward(self, x, gradx=None, pos=None, grid=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''
        x_latent = []
        x_fts = []

        if gradx.ndim == x.ndim and self.add_grad_channel:
            x = torch.cat([x, gradx], dim=-1)

        if self.spacial_fc:
            x = torch.cat([x, grid], dim=-1)
            x = self.fc(x)

        for layer in self.spectral_conv:
            if self.return_freq:
                x, x_ft = layer(x)
                x_fts.append(x_ft.contiguous())
            else:
                x = layer(x)

            if self.return_latent:
                x_latent.append(x.contiguous())

        x = self.regressor(x)

        if self.normalizer:
            x = self.normalizer.inverse_transform(x)

        if self.return_freq or self.return_latent:
            return dict(preds=x,
                preds_latent=x_latent,
                preds_freq=x_fts)
        else:
            return dict(preds=x)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = dict(in_dim=3,
                n_hidden=64, # dmodel of the input for spectral conv
                freq_dim=64,  # number of frequency features
                out_dim=1,
                modes=16,  # number of fourier modes
                num_spectral_layers=8,
                n_grid=201,
                dim_feedforward=None,
                spacial_dim=2,
                spacial_fc=True,
                return_freq=True,  # to be consistent with trainer
                activation='silu',
                last_activation=False,
                add_grad_channel=True,
                dropout=0)

    model = FourierNeuralOperator(**config)
    model.to(device)
    batch_size, n_grid = 8, 201
    summary(model, input_size=[(batch_size, n_grid, n_grid, 1),
                               (batch_size, n_grid, n_grid, 2),
                               (batch_size, 1, 1),
                               (batch_size, n_grid, n_grid, 2)], device=device)
