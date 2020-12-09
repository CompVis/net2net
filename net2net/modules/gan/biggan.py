# built upon the very nice https://github.com/LoreGoetschalckx/GANalyze
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d
from torch.nn import Parameter
import torch.nn.functional as F

from net2net.modules.autoencoder.basic import ActNorm
from net2net.ckpt_util import get_ckpt_path


class GANException(Exception):
    pass


def l2normalize(v, eps=1e-4):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        _w = w.view(height, -1)
        for _ in range(self.power_iterations):
            v = l2normalize(torch.matmul(_w.t(), u))
            u = l2normalize(torch.matmul(_w, v))

        sigma = u.dot((_w).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation=F.relu):
        super().__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.theta = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False))
        self.phi = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False))
        self.pool = nn.MaxPool2d(2, 2)
        self.g = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1, bias=False))
        self.o_conv = SpectralNorm(nn.Conv2d(in_channels=in_dim // 2, out_channels=in_dim, kernel_size=1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        N = height * width

        theta = self.theta(x)
        phi = self.phi(x)
        phi = self.pool(phi)
        phi = phi.view(m_batchsize, -1, N // 4)
        theta = theta.view(m_batchsize, -1, N)
        theta = theta.permute(0, 2, 1)
        attention = self.softmax(torch.bmm(theta, phi))  # BX (N) X (N)
        g = self.pool(self.g(x)).view(m_batchsize, -1, N // 4)
        attn_g = torch.bmm(g, attention.permute(0, 2, 1)).view(m_batchsize, -1, width, height)
        out = self.o_conv(attn_g)
        return self.gamma * out + x


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = BatchNorm2d(num_features, affine=False, eps=1e-4)
        self.gamma_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False))
        self.beta_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False))

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma_embed(y) + 1
        beta = self.beta_embed(y)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class ConditionalActNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = ActNorm(num_features)
        self.gamma_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False))
        self.beta_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False))

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma_embed(y) + 1
        beta = self.beta_embed(y)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class BatchNorm2dWrap(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = BatchNorm2d(*args, **kwargs)

    def forward(self, x, y=None):
        return self.bn(x)


class ActNorm2dWrap(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = ActNorm(*args, **kwargs)

    def forward(self, x, y=None):
        return self.bn(x)


def update_G_linear(biggan_generator, n_in, n_out=16*16*96):
    biggan_generator.G_linear = SpectralNorm(nn.Linear(n_in, n_out))
    return biggan_generator


class GBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=[3, 3],
        padding=1,
        stride=1,
        n_class=None,
        bn=True,
        activation=F.relu,
        upsample=True,
        downsample=False,
        z_dim=148,
        use_actnorm=False,
        conditional=True
    ):
        super().__init__()

        self.conv0 = SpectralNorm(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=True if bn or use_actnorm else True)
        )
        self.conv1 = SpectralNorm(
            nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding, bias=True if bn or use_actnorm else True)
        )

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))
            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            if conditional:
                self.HyperBN = ConditionalBatchNorm2d(in_channel, z_dim)
                self.HyperBN_1 = ConditionalBatchNorm2d(out_channel, z_dim)
            else:
                self.HyperBN = BatchNorm2dWrap(in_channel, z_dim)
                self.HyperBN_1 = BatchNorm2dWrap(out_channel, z_dim)
        else:
            if use_actnorm:
                if conditional:
                    self.HyperBN = ConditionalActNorm2d(in_channel, z_dim)
                    self.HyperBN_1 = ConditionalActNorm2d(out_channel, z_dim)
                else:
                    self.HyperBN = ActNorm2dWrap(in_channel)
                    self.HyperBN_1 = ActNorm2dWrap(out_channel)

    def forward(self, input, condition=None):
        out = input

        if self.bn:
            out = self.HyperBN(out, condition)
        out = self.activation(out)
        # return out
        if self.upsample:
            # different form papers
            out = F.interpolate(out, scale_factor=2)
        out = self.conv0(out)
        if self.bn:
            out = self.HyperBN_1(out, condition)
        out = self.activation(out)
        out = self.conv1(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        if self.skip_proj:
            skip = input
            if self.upsample:
                # different form papers
                skip = F.interpolate(skip, scale_factor=2)
            skip = self.conv_sc(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)
        else:
            skip = input
        return out + skip


class Generator128(nn.Module):
    def __init__(self, code_dim=120, n_class=1000, chn=96, debug=False, use_actnorm=False):
        super().__init__()

        self.linear = nn.Linear(n_class, 128, bias=False)

        if debug:
            chn = 8

        self.first_view = 16 * chn
        self.G_linear = SpectralNorm(nn.Linear(20, 4 * 4 * 16 * chn))
        z_dim = code_dim + 28

        self.GBlock = nn.ModuleList([
            GBlock(16 * chn, 16 * chn, n_class=n_class, z_dim=z_dim),
            GBlock(16 * chn, 8 * chn, n_class=n_class, z_dim=z_dim),
            GBlock(8 * chn, 4 * chn, n_class=n_class, z_dim=z_dim),
            GBlock(4 * chn, 2 * chn, n_class=n_class, z_dim=z_dim),
            GBlock(2 * chn, 1 * chn, n_class=n_class, z_dim=z_dim),
        ])

        self.sa_id = 4
        self.num_split = len(self.GBlock) + 1
        self.attention = SelfAttention(2 * chn)
        if not use_actnorm:
            self.ScaledCrossReplicaBN = BatchNorm2d(1 * chn, eps=1e-4)
        else:
            self.ScaledCrossReplicaBN = ActNorm(1 * chn)
        self.colorize = SpectralNorm(nn.Conv2d(1 * chn, 3, [3, 3], padding=1))


    def forward(self, input, class_id, from_class_embedding=False):
        codes = torch.chunk(input, self.num_split, 1)
        if from_class_embedding:
            class_emb = class_id  # 128
        else:
            class_emb = self.linear(class_id)  # 128

        out = self.G_linear(codes[0])
        out = out.view(-1, 4, 4, self.first_view).permute(0, 3, 1, 2)
        for i, (code, GBlock) in enumerate(zip(codes[1:], self.GBlock)):
            if i == self.sa_id:
                out = self.attention(out)
            condition = torch.cat([code, class_emb], 1)
            out = GBlock(out, condition)

        out = self.ScaledCrossReplicaBN(out)
        out = F.relu(out)
        out = self.colorize(out)
        return torch.tanh(out)

    def encode(self, *args, **kwargs):
        raise GANException("Sorry, I'm a GAN and not very helpful for encoding.")


    def decode(self, z, cls):
        z = z.float()
        cls_one_hot = torch.nn.functional.one_hot(cls, num_classes=1000).float()
        return self.forward(z, cls_one_hot)


class VariableDimGenerator128(Generator128):
    """splits latent code z of dimension d in sizes (d-(k-1)*20, 20, 20, ..., 20),
    here; k=5 (?), k is number of GBlocks
    use extra_z_dims to add extra dimensions to z into an already trained
    generator
    """
    def __init__(self, code_dim, *args, extra_z_dims=list(), **kwargs):
        super().__init__(*args, **kwargs)
        first_split = code_dim - (self.num_split-1)*20
        self.split_at = [first_split] + [20 for i in range(self.num_split-1)]
        self.extra_z_dims = extra_z_dims
        self.extra_linears = nn.ModuleList()
        for extra_z_dim in self.extra_z_dims:
            self.extra_linears.append(SpectralNorm(nn.Linear(extra_z_dim, 4*4*self.first_view)))
            self.split_at += [extra_z_dim]

    def forward(self, input, class_id):
        codes = torch.split(input, self.split_at, 1)
        class_emb = self.linear(class_id)  # 128

        out = self.G_linear(codes[0])

        if self.extra_z_dims:
            extra_codes = codes[self.num_split:]
            for extra_code, extra_linear in zip(extra_codes, self.extra_linears):
                out = out + extra_linear(extra_code)

        out = out.view(-1, 4, 4, self.first_view).permute(0, 3, 1, 2)
        for i, (code, GBlock) in enumerate(zip(codes[1:], self.GBlock)):
            if i == self.sa_id:
                out = self.attention(out)
            condition = torch.cat([code, class_emb], 1)
            out = GBlock(out, condition)

        out = self.ScaledCrossReplicaBN(out)
        out = F.relu(out)
        out = self.colorize(out)
        return torch.tanh(out)


class Generator256(nn.Module):
    def __init__(self, code_dim=140, n_class=1000, chn=96, debug=False, use_actnorm=False):
        super().__init__()
        self.linear = nn.Linear(n_class, 128, bias=False)

        if debug:
            chn = 8
        self.first_view = 16 * chn
        self.G_linear = SpectralNorm(nn.Linear(20, 4 * 4 * 16 * chn))

        self.GBlock = nn.ModuleList([
            GBlock(16 * chn, 16 * chn, n_class=n_class, use_actnorm=use_actnorm),
            GBlock(16 * chn, 8 * chn, n_class=n_class, use_actnorm=use_actnorm),
            GBlock(8 * chn, 8 * chn, n_class=n_class, use_actnorm=use_actnorm),
            GBlock(8 * chn, 4 * chn, n_class=n_class, use_actnorm=use_actnorm),
            GBlock(4 * chn, 2 * chn, n_class=n_class, use_actnorm=use_actnorm),
            GBlock(2 * chn, 1 * chn, n_class=n_class, use_actnorm=use_actnorm),
        ])

        self.sa_id = 5
        self.num_split = len(self.GBlock) + 1
        self.attention = SelfAttention(2 * chn)
        if not use_actnorm:
            self.ScaledCrossReplicaBN = BatchNorm2d(1 * chn, eps=1e-4)
        else:
            self.ScaledCrossReplicaBN = ActNorm(1 * chn)
        self.colorize = SpectralNorm(nn.Conv2d(1 * chn, 3, [3, 3], padding=1))


    def forward(self, input, class_id, from_class_embedding=False):
        codes = torch.chunk(input, self.num_split, 1)
        if from_class_embedding:
            class_emb = class_id  # 128
        else:
            class_emb = self.linear(class_id)  # 128
        out = self.G_linear(codes[0])
        out = out.view(-1, 4, 4, self.first_view).permute(0, 3, 1, 2)
        for i, (code, GBlock) in enumerate(zip(codes[1:], self.GBlock)):
            if i == self.sa_id:
                out = self.attention(out)
            condition = torch.cat([code, class_emb], 1)
            out = GBlock(out, condition)

        out = self.ScaledCrossReplicaBN(out)
        out = F.relu(out)
        out = self.colorize(out)
        return torch.tanh(out)

    def encode(self, *args, **kwargs):
        raise GANException("Sorry, I'm a GAN and not very helpful for encoding.")

    def decode(self, z, cls):
        z = z.float()
        cls_one_hot = torch.nn.functional.one_hot(cls, num_classes=1000).float()
        return self.forward(z, cls_one_hot)


class VariableDimGenerator256(Generator256):
    """splits latent code z of dimension d in sizes (d-(k-1)*20, 20, 20, ..., 20),
    here; k=6 (?), k is number of GBlocks
    use extra_z_dims to add extra dimensions to z into an already trained
    generator
    """
    def __init__(self, code_dim, *args, extra_z_dims=list(), **kwargs):
        super().__init__(*args, **kwargs)
        first_split = code_dim - (self.num_split-1)*20
        self.split_at = [first_split] + [20 for i in range(self.num_split-1)]
        self.extra_z_dims = extra_z_dims
        self.extra_linears = nn.ModuleList()
        for extra_z_dim in self.extra_z_dims:
            self.extra_linears.append(SpectralNorm(nn.Linear(extra_z_dim, 4*4*self.first_view)))
            self.split_at += [extra_z_dim]

    def forward(self, input, class_id):
        codes = torch.split(input, self.split_at, 1)
        class_emb = self.linear(class_id)  # 128

        out = self.G_linear(codes[0])

        if self.extra_z_dims:
            extra_codes = codes[self.num_split:]
            for extra_code, extra_linear in zip(extra_codes, self.extra_linears):
                out = out + extra_linear(extra_code)

        out = out.view(-1, 4, 4, self.first_view).permute(0, 3, 1, 2)
        for i, (code, GBlock) in enumerate(zip(codes[1:], self.GBlock)):
            if i == self.sa_id:
                out = self.attention(out)
            condition = torch.cat([code, class_emb], 1)
            out = GBlock(out, condition)

        out = self.ScaledCrossReplicaBN(out)
        out = F.relu(out)
        out = self.colorize(out)
        return torch.tanh(out)


class BigGANWrapper(nn.Module):
    def __init__(self, image_size=256):
        super().__init__()
        self.class_embedding_dim = 1000
        self.decoder = load_generator(image_size, pretrained=True, use_actnorm=False,
                                      n_class=self.class_embedding_dim)

    def forward(self, x, labels, labels_are_one_hot=False):
        if not labels_are_one_hot:
            one_hot = torch.nn.functional.one_hot(labels, num_classes=self.class_embedding_dim).float()
        else:
            one_hot = labels
            if labels.shape[1] != self.class_embedding_dim:
                zeros = torch.zeros(labels.shape[0], self.class_embedding_dim).to(labels)
                zeros[:,:labels.shape[1]] = labels
                one_hot = zeros
        x = self.decoder(x, one_hot)
        return x

    def embed_labels(self, labels, labels_are_one_hot=False):
        """embeds labels, usually in a 128-dim space"""
        if not labels_are_one_hot:
            one_hot = torch.nn.functional.one_hot(labels, num_classes=self.class_embedding_dim).float()
        else:
            one_hot = labels
        return self.decoder.linear(one_hot)

    def generate_from_embedding(self, x, class_emb):
        return self.decoder(x, class_emb, from_class_embedding=True)


def load_generator(size, pretrained=True, use_actnorm=False, n_class=1000):
    """ size an integer in [128, 256]"""
    assert size in [128, 256]
    __generators = {128: Generator128, 256: Generator256}
    __names = {128: "biggan_128", 256: "biggan_256"}
    if pretrained:
        assert n_class==1000
    G = __generators[size](use_actnorm=use_actnorm, n_class=n_class)
    ckpt = get_ckpt_path(__names[size], "net2net/modules/gan/pretrained_biggan")
    G.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
    return G


def load_variable_latsize_generator(size, z_dim,
                                    n_class = 1000,
                                    use_actnorm=False,
                                    extra_z_dims=list()
                                    ):
    generators = {128: VariableDimGenerator128, 256: VariableDimGenerator256}
    G = generators[size](z_dim, use_actnorm=use_actnorm, n_class=n_class,
                         extra_z_dims=extra_z_dims)

    split_sizes = {128: 5*20, 256: 6*20}
    G = update_G_linear(G, z_dim - split_sizes[size])  # add new trainable layer to adopt for variable z_dim size
    return G


if __name__ == "__main__":
    G = load_generator(256)
    G = load_generator(128)
    print("done.")