import torch
import torch.nn as nn


def nll(sample):
    if len(sample.shape) == 2:
        sample = sample[:,:,None,None]
    return 0.5*torch.sum(torch.pow(sample, 2), dim=[1,2,3])


class NLL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample, logdet, split="train"):
        nll_loss = torch.mean(nll(sample))
        assert len(logdet.shape) == 1
        nlogdet_loss = -torch.mean(logdet)
        loss = nll_loss + nlogdet_loss
        reference_nll_loss = torch.mean(nll(torch.randn_like(sample)))
        log = {f"{split}/total_loss": loss, f"{split}/reference_nll_loss": reference_nll_loss,
               f"{split}/nlogdet_loss": nlogdet_loss, f"{split}/nll_loss": nll_loss,
               }
        return loss, log