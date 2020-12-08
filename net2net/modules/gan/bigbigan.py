import torch
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow_hub as hub


class BigBiGAN(object):
    def __init__(self,
                 module_path='https://tfhub.dev/deepmind/bigbigan-resnet50/1',
                 allow_growth=True):
        """Initialize a BigBiGAN from the given TF Hub module."""
        self._module = hub.Module(module_path)

        # encode graph
        self.enc_ph = self.make_encoder_ph()
        self.z_sample = self.encode_graph(self.enc_ph)
        self.z_mean = self.encode_graph(self.enc_ph, return_all_features=True)['z_mean']

        # decode graph
        self.gen_ph = self.make_generator_ph()
        self.gen_samples = self.generate_graph(self.gen_ph, upsample=True)

        # session
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(allow_growth=allow_growth)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(init)

    def generate_graph(self, z, upsample=False):
        """Run a batch of latents z through the generator to generate images.

        Args:
            z: A batch of 120D Gaussian latents, shape [N, 120].

        Returns: a batch of generated RGB images, shape [N, 128, 128, 3], range
            [-1, 1].
        """
        outputs = self._module(z, signature='generate', as_dict=True)
        return outputs['upsampled' if upsample else 'default']

    def make_generator_ph(self):
        """Creates a tf.placeholder with the dtype & shape of generator inputs."""
        info = self._module.get_input_info_dict('generate')['z']
        return tf.placeholder(dtype=info.dtype, shape=info.get_shape())

    def encode_graph(self, x, return_all_features=False):
        """Run a batch of images x through the encoder.

        Args:
            x: A batch of data (256x256 RGB images), shape [N, 256, 256, 3], range
                [-1, 1].
            return_all_features: If True, return all features computed by the encoder.
                Otherwise (default) just return a sample z_hat.

        Returns: the sample z_hat of shape [N, 120] (or a dict of all features if
            return_all_features).
        """
        outputs = self._module(x, signature='encode', as_dict=True)
        return outputs if return_all_features else outputs['z_sample']

    def make_encoder_ph(self):
        """Creates a tf.placeholder with the dtype & shape of encoder inputs."""
        info = self._module.get_input_info_dict('encode')['x']
        return tf.placeholder(dtype=info.dtype, shape=info.get_shape())

    @torch.no_grad()
    def encode(self, x_torch):
        x_np = x_torch.detach().permute(0,2,3,1).cpu().numpy()
        feed_dict = {self.enc_ph: x_np}
        z = self.sess.run(self.z_sample, feed_dict=feed_dict)
        z_torch = torch.tensor(z).to(device=x_torch.device)
        return z_torch.unsqueeze(-1).unsqueeze(-1)

    @torch.no_grad()
    def decode(self, z_torch):
        z_np = z_torch.detach().squeeze(-1).squeeze(-1).cpu().numpy()
        feed_dict = {self.gen_ph: z_np}
        x = self.sess.run(self.gen_samples, feed_dict=feed_dict)
        x = x.transpose(0,3,1,2)
        x_torch = torch.tensor(x).to(device=z_torch.device)
        return x_torch

    def eval(self):
        # interface requirement
        return self
