import torch
import torch.nn as nn

from net2net.modules.gan.biggan import load_variable_latsize_generator

class ClassUp(nn.Module):
    def __init__(self, dim, depth, hidden_dim=256, use_sigmoid=False, out_dim=None):
        super().__init__()
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.LeakyReLU())
        for d in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim, dim if out_dim is None else out_dim))
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x.squeeze(-1).squeeze(-1))
        x = torch.nn.functional.softmax(x, dim=1)
        return x


class BigGANDecoderWrapper(nn.Module):
    """Wraps a BigGAN into our autoencoding framework"""
    def __init__(self, z_dim, in_size=128, use_actnorm_in_dec=False, extra_z_dims=list()):
        super().__init__()
        self.z_dim = z_dim
        class_embedding_dim = 1000
        self.extra_z_dims = extra_z_dims
        self.map_to_class_embedding = ClassUp(z_dim, depth=2, hidden_dim=2*class_embedding_dim,
                                              use_sigmoid=False, out_dim=class_embedding_dim)
        self.decoder = load_variable_latsize_generator(in_size, z_dim,
                                                       use_actnorm=use_actnorm_in_dec,
                                                       n_class=class_embedding_dim,
                                                       extra_z_dims=self.extra_z_dims)

    def forward(self, x, labels=None):
        emb = self.map_to_class_embedding(x[:,:self.z_dim,...])
        x = self.decoder(x, emb)
        return x