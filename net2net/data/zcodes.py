import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class PRNGMixin(object):
    """Adds a prng property which is a numpy RandomState which gets
    reinitialized whenever the pid changes to avoid synchronized sampling
    behavior when used in conjunction with multiprocessing."""

    @property
    def prng(self):
        currentpid = os.getpid()
        if getattr(self, "_initpid", None) != currentpid:
            self._initpid = currentpid
            self._prng = np.random.RandomState()
        return self._prng


class TrainSamples(Dataset, PRNGMixin):
    def __init__(self, n_samples, z_shape, n_classes, truncation=0):
        self.n_samples = n_samples
        self.z_shape = z_shape
        self.n_classes = n_classes
        self.truncation_threshold = truncation
        if self.truncation_threshold > 0:
            print("Applying truncation at level {}".format(self.truncation_threshold))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        z = self.prng.randn(*self.z_shape)
        if self.truncation_threshold > 0:
            for k, zi in enumerate(z):
                while abs(zi) > self.truncation_threshold:
                    zi = self.prng.randn(1)
                z[k] = zi
        cls = self.prng.randint(self.n_classes)
        return {"z": z.astype(np.float32), "class": cls}


class TestSamples(Dataset):
    def __init__(self, n_samples, z_shape, n_classes, truncation=0):
        self.prng = np.random.RandomState(1)
        self.n_samples = n_samples
        self.z_shape = z_shape
        self.n_classes = n_classes
        self.truncation_threshold = truncation
        if self.truncation_threshold > 0:
            print("Applying truncation at level {}".format(self.truncation_threshold))
        self.zs = self.prng.randn(self.n_samples, *self.z_shape)
        if self.truncation_threshold > 0:
            print("Applying truncation at level {}".format(self.truncation_threshold))
            ix = 0
            for z in tqdm(self.zs, desc="Truncation:"):
                for k, zi in enumerate(z):
                    while abs(zi) > self.truncation_threshold:
                        zi = self.prng.randn(1)
                    z[k] = zi
                self.zs[ix] = z
                ix += 1
            print("Created truncated test data.")
        self.clss = self.prng.randint(self.n_classes, size=(self.n_samples,))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return {"z": self.zs[i].astype(np.float32), "class": self.clss[i]}


class RestrictedTrainSamples(Dataset, PRNGMixin):
    def __init__(self, n_samples, z_shape, truncation=0):
        index_path = "data/coco_imagenet_overlap_idx.txt"
        self.n_samples = n_samples
        self.z_shape = z_shape
        self.classes = np.loadtxt(index_path).astype(int)
        self.truncation_threshold = truncation
        if self.truncation_threshold > 0:
            print("Applying truncation at level {}".format(self.truncation_threshold))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        z = self.prng.randn(*self.z_shape)
        if self.truncation_threshold > 0:
            for k, zi in enumerate(z):
                while abs(zi) > self.truncation_threshold:
                    zi = self.prng.randn(1)
                z[k] = zi
        cls = self.prng.choice(self.classes)
        return {"z": z.astype(np.float32), "class": cls}


class RestrictedTestSamples(Dataset):
    def __init__(self, n_samples, z_shape, truncation=0):
        index_path = "data/coco_imagenet_overlap_idx.txt"

        self.prng = np.random.RandomState(1)
        self.n_samples = n_samples
        self.z_shape = z_shape

        self.classes = np.loadtxt(index_path).astype(int)
        self.clss = self.prng.choice(self.classes, size=(self.n_samples,), replace=True)
        self.truncation_threshold = truncation
        self.zs = self.prng.randn(self.n_samples, *self.z_shape)
        if self.truncation_threshold > 0:
            print("Applying truncation at level {}".format(self.truncation_threshold))
            ix = 0
            for z in tqdm(self.zs, desc="Truncation:"):
                for k, zi in enumerate(z):
                    while abs(zi) > self.truncation_threshold:
                        zi = self.prng.randn(1)
                    z[k] = zi
                self.zs[ix] = z
                ix += 1
            print("Created truncated test data.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return {"z": self.zs[i].astype(np.float32), "class": self.clss[i]}


