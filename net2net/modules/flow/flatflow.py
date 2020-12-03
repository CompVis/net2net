import torch
import torch.nn as nn
import numpy as np

from net2net.modules.autoencoder.basic import ActNorm
from net2net.modules.flow.blocks import UnconditionalFlatDoubleCouplingFlowBlock, PureAffineDoubleCouplingFlowBlock, \
    ConditionalFlatDoubleCouplingFlowBlock
from net2net.modules.flow.base import NormalizingFlow
from net2net.modules.autoencoder.basic import FeatureLayer, DenseEncoderLayer
from net2net.modules.flow.blocks import BasicFullyConnectedNet


class UnconditionalFlatCouplingFlow(NormalizingFlow):
    """Flat, multiple blocks of ActNorm, DoubleAffineCoupling, Shuffle"""
    def __init__(self, in_channels, n_flows, hidden_dim, hidden_depth):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.sub_layers = nn.ModuleList()

        for flow in range(self.n_flows):
            self.sub_layers.append(UnconditionalFlatDoubleCouplingFlowBlock(
                                   self.in_channels, self.mid_channels,
                                   self.num_blocks)
                                   )

    def forward(self, x, reverse=False):
        if len(x.shape) == 2:
            x = x[:,:,None,None]
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, reverse=True)
            return x

    def reverse(self, out):
        if len(out.shape) == 2:
            out = out[:,:,None,None]
        return self(out, reverse=True)

    def sample(self, num_samples, device="cpu"):
        zz = torch.randn(num_samples, self.in_channels, 1, 1).to(device)
        return self.reverse(zz)

    def get_last_layer(self):
        return getattr(self.sub_layers[-1].coupling.t[-1].main[-1], 'weight')


class PureAffineFlatCouplingFlow(UnconditionalFlatCouplingFlow):
    """Flat, multiple blocks of DoubleAffineCoupling"""
    def __init__(self, in_channels, n_flows, hidden_dim, hidden_depth):
        super().__init__(in_channels, n_flows, hidden_dim, hidden_depth)
        del self.sub_layers
        self.sub_layers = nn.ModuleList()
        for flow in range(self.n_flows):
            self.sub_layers.append(PureAffineDoubleCouplingFlowBlock(
                                   self.in_channels, self.mid_channels,
                                   self.num_blocks)
            )


class DenseEmbedder(nn.Module):
    """Supposed to map small-scale features (e.g. labels) to some given latent dim"""
    def __init__(self, in_dim, up_dim, depth=4, given_dims=None):
        super().__init__()
        self.net = nn.ModuleList()
        if given_dims is not None:
            assert given_dims[0] == in_dim
            assert given_dims[-1] == up_dim
            dims = given_dims
        else:
            dims = np.linspace(in_dim, up_dim, depth).astype(int)
        for l in range(len(dims)-2):
            self.net.append(nn.Conv2d(dims[l], dims[l + 1], 1))
            self.net.append(ActNorm(dims[l + 1]))
            self.net.append(nn.LeakyReLU(0.2))

        self.net.append(nn.Conv2d(dims[-2], dims[-1], 1))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x.squeeze(-1).squeeze(-1)


class Embedder(nn.Module):
    """Embeds a 4-dim tensor onto dense latent code, much like the classic encoder."""
    def __init__(self, in_spatial_size, in_channels, emb_dim, n_down=4):
        super().__init__()
        self.feature_layers = nn.ModuleList()
        norm = 'an'  # hard coded yes
        bottleneck_size = in_spatial_size // 2**n_down
        self.feature_layers.append(FeatureLayer(0, in_channels=in_channels, norm=norm))
        for scale in range(1, n_down):
            self.feature_layers.append(FeatureLayer(scale, norm=norm))
        self.dense_encode = DenseEncoderLayer(n_down, bottleneck_size, emb_dim)
        if n_down == 1:
            # add some extra parameters to make model a little more powerful ?
            print(" Warning: Embedder for ConditionalTransformer has only one down-sampling step. You might want to "
                  "increase its capacity.")

    def forward(self, input):
        h = input
        for layer in self.feature_layers:
            h = layer(h)
        h = self.dense_encode(h)
        return h.squeeze(-1).squeeze(-1)


class ConditionalFlatCouplingFlow(nn.Module):
    """Flat version. Feeds an embedding into the flow in every block"""
    def __init__(self, in_channels, conditioning_dim, embedding_dim, hidden_dim, hidden_depth,
                 n_flows, conditioning_option="none", activation='lrelu',
                 conditioning_hidden_dim=256, conditioning_depth=2, conditioner_use_bn=False,
                 conditioner_use_an=False):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = embedding_dim
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.conditioning_option = conditioning_option
        # TODO: also for spatial inputs...
        if conditioner_use_bn:
            assert not conditioner_use_an, 'Can not use ActNorm and BatchNorm simultaneously in Embedder.'
            print("Note: Conditioning network uses batch-normalization. "
                  "Make sure to train with a sufficiently large batch size")

        self.embedder = BasicFullyConnectedNet(dim=conditioning_dim,
                                               depth=conditioning_depth,
                                               out_dim=embedding_dim,
                                               hidden_dim=conditioning_hidden_dim,
                                               use_bn=conditioner_use_bn,
                                               use_an=conditioner_use_an)

        self.sub_layers = nn.ModuleList()
        if self.conditioning_option.lower() != "none":
            self.conditioning_layers = nn.ModuleList()
        for flow in range(self.n_flows):
            self.sub_layers.append(ConditionalFlatDoubleCouplingFlowBlock(
                self.in_channels, self.cond_channels, self.mid_channels,
                self.num_blocks, activation=activation)
            )
            if self.conditioning_option.lower() != "none":
                self.conditioning_layers.append(nn.Conv2d(self.cond_channels, self.cond_channels, 1))

    def forward(self, x, cond, reverse=False):
        hconds = list()
        if len(cond.shape) == 4:
            if cond.shape[2] == 1:
                assert cond.shape[3] == 1
                cond = cond.squeeze(-1).squeeze(-1)
            else:
                raise ValueError("Spatial conditionings not yet supported. TODO")
        embedding = self.embedder(cond.float())
        hcond = embedding[:, :, None, None]
        for i in range(self.n_flows):
            if self.conditioning_option.lower() == "parallel":
                hcond = self.conditioning_layers[i](embedding)
            elif self.conditioning_option.lower() == "sequential":
                hcond = self.conditioning_layers[i](hcond)
            hconds.append(hcond)
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x, hconds[i])
                logdet = logdet + logdet_
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, hconds[i], reverse=True)
            return x

    def reverse(self, out, xcond):
        return self(out, xcond, reverse=True)

    def sample(self, xc):
        zz = torch.randn(xc.shape[0], self.in_channels, 1, 1).to(xc)
        return self.reverse(zz, xc)


