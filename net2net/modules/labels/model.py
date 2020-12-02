import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Labelator(nn.Module):
    def __init__(self, num_classes, as_one_hot=True):
        super().__init__()
        self.num_classes = num_classes
        self.as_one_hot = as_one_hot

    def encode(self, x):
        if self.as_one_hot:
            x = self.make_one_hot(x)
        return x

    def other_label(self, given_label):
        # if only two classes are present, inverts them
        others = []
        for l in given_label:
            other = int(np.random.choice(np.arange(self.num_classes)))
            while other == l:
                other = int(np.random.choice(np.arange(self.num_classes)))
            others.append(other)
        return torch.LongTensor(others)

    def make_one_hot(self, label):
        one_hot = F.one_hot(label, num_classes=self.num_classes)
        return one_hot
