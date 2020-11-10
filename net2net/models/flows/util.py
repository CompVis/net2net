import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.neighbors import KernelDensity


def kde2D(x, y, bandwidth, xbins=250j, ybins=250j, **kwargs):
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins,
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


def plot2d(x, savepath=None):
    """make a scatter plot of x and return an Image of it"""
    x = x.cpu().numpy().squeeze()
    fig = plt.figure(dpi=300)
    xx, yy, zz = kde2D(x[:,0], x[:,1], 0.1)
    plt.pcolormesh(xx, yy, zz)
    plt.scatter(x[:,0], x[:, 1], s=0.1, c='mistyrose')
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    return fig


def reshape_to_grid(x, num_samples=16, iw=28, ih=28, nc=1):
    x = x[:num_samples]
    x = x.detach().cpu()
    x = torch.reshape(x, (x.shape[0], nc, iw, ih))
    xgrid = torchvision.utils.make_grid(x)
    return xgrid