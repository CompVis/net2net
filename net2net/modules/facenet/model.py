import torch
import torch.nn as nn
from torch.nn import functional as F

from net2net.modules.facenet.inception_resnet_v1 import InceptionResnetV1


"""FaceNet adopted from https://github.com/timesler/facenet-pytorch"""


class FaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        # InceptionResnetV1 has a bottleneck of size 512
        self.net = InceptionResnetV1(pretrained='vggface2').eval()

    def _pre_process(self, x):
        # TODO: neccessary for InceptionResnetV1?
        # seems like mtcnn (multi-task cnn) preprocessing is neccessary, but not 100% sure
        return x

    def forward(self, x, return_logits=False):
        # output are logits of size 8631 or embeddings of size 512
        x = self._pre_process(x)
        emb = self.net(x)
        if return_logits:
            return self.net.logits(emb)
        return emb

    def encode(self, x):
        return self(x)

    def return_features(self, x):
        """ returned features have the following dimensions:

             torch.Size([11, 3, 128, 128]),  x   49152
             torch.Size([11, 192, 28, 28]),  x   150528
             torch.Size([11, 896, 6, 6]),    x   32256
             torch.Size([11, 1792, 1, 1]),   x   1792
             torch.Size([11, 512])           x   512
             logits (8xxx)                   x   8xxx
        """

        x = self._pre_process(x)
        features = [x]   # this
        x = self.net.conv2d_1a(x)
        x = self.net.conv2d_2a(x)
        x = self.net.conv2d_2b(x)
        x = self.net.maxpool_3a(x)
        x = self.net.conv2d_3b(x)
        x = self.net.conv2d_4a(x)
        features.append(x)  # this
        x = self.net.conv2d_4b(x)
        x = self.net.repeat_1(x)
        x = self.net.mixed_6a(x)
        features.append(x)  # this
        x = self.net.repeat_2(x)
        x = self.net.mixed_7a(x)
        x = self.net.repeat_3(x)
        x = self.net.block8(x)
        x = self.net.avgpool_1a(x)
        features.append(x)  # this
        x = self.net.dropout(x)
        x = self.net.last_linear(x.view(x.shape[0], -1))
        x = self.net.last_bn(x)
        emb = F.normalize(x, p=2, dim=1)  # the final embeddings
        features.append(emb[..., None, None])  # need extra dimensions for flow later
        features.append(self.net.logits(emb).unsqueeze(-1).unsqueeze(-1))
        return features  # has 6 elements as of now

