import torch.nn as nn


class NormalizingFlow(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        # return transformed, logdet
        raise NotImplementedError

    def reverse(self, *args, **kwargs):
        # return transformed_reverse
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        # return sample
        raise NotImplementedError
