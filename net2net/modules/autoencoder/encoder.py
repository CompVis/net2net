import torch
import torch.nn as nn
import torchvision
from torchvision import models

from net2net.modules.autoencoder.basic import ActNorm, DenseEncoderLayer


class ResnetEncoder(nn.Module):
    def __init__(self, z_dim, in_size, in_channels=3,
                 pretrained=False, type="resnet50",
                 double_z=True, pre_process=True,
                 ):
        super().__init__()
        __possible_resnets = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101
        }
        self.use_preprocess = pre_process
        self.in_channels = in_channels
        norm_layer = ActNorm
        self.z_dim = z_dim
        self.model = __possible_resnets[type](pretrained=pretrained, norm_layer=norm_layer)

        self.image_transform = torchvision.transforms.Compose(
                [torchvision.transforms.Lambda(self.normscale)]
                )

        size_pre_fc = self.get_spatial_size(in_size)
        assert size_pre_fc[2]==size_pre_fc[3], 'Output spatial size is not quadratic'
        spatial_size = size_pre_fc[2]
        num_channels_pre_fc = size_pre_fc[1]
        # replace last fc
        self.model.fc = DenseEncoderLayer(0,
                                          spatial_size=spatial_size,
                                          out_size=2*z_dim if double_z else z_dim,
                                          in_channels=num_channels_pre_fc)
        if self.in_channels != 3:
            self.model.in_ch_match = nn.Conv2d(self.in_channels, 3, 3, 1)

    def forward(self, x):
        if self.use_preprocess:
            x = self.pre_process(x)
        if self.in_channels != 3:
            assert not self.use_preprocess
            x = self.model.in_ch_match(x)
        features = self.features(x)
        encoding = self.model.fc(features)
        return encoding

    def rescale(self, x):
        return 0.5 * (x + 1)

    def normscale(self, image):
        normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        return torch.stack([normalize(self.rescale(x)) for x in image])

    def features(self, x):
        if self.use_preprocess:
            x = self.pre_process(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        return x

    def post_features(self, x):
        x = self.model.fc(x)
        return x

    def pre_process(self, x):
        x = self.image_transform(x)
        return x

    def get_spatial_size(self, ipt_size):
        x = torch.randn(1, 3, ipt_size, ipt_size)
        return self.features(x).size()

    @property
    def mean(self):
        return [0.485, 0.456, 0.406]

    @property
    def std(self):
        return [0.229, 0.224, 0.225]

    @property
    def input_size(self):
        return [3, 224, 224]
