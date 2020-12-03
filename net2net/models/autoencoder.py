import torch
import pytorch_lightning as pl

from net2net.modules.distributions.distributions import DiagonalGaussianDistribution
from translation import instantiate_from_config


class BigAE(pl.LightningModule):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 loss_config,
                 ckpt_path=None,
                 ignore_keys=[]
                 ):
        super().__init__()
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.loss = instantiate_from_config(loss_config)

        if ckpt_path is not None:
            print("Loading model from {}".format(ckpt_path))
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        try:
            sd = torch.load(path, map_location="cpu")["state_dict"]
        except KeyError:
            sd = torch.load(path, map_location="cpu")

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if len(missing) > 0:
            print(f"Missing keys in state dict: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys in state dict: {unexpected}")

    def encode(self, x, return_mode=False):
        moments = self.encoder(x)
        posterior = DiagonalGaussianDistribution(moments, deterministic=False)
        if return_mode:
            return posterior.mode()
        return posterior.sample()

    def decode(self, z):
        if len(z.shape) == 4:
            z = z.squeeze(-1).squeeze(-1)
        return self.decoder(z)

    def forward(self, x):
        moments = self.encoder(x)
        posterior = DiagonalGaussianDistribution(moments)
        h = posterior.sample()
        reconstructions = self.decoder(h.squeeze(-1).squeeze(-1))
        return reconstructions, posterior

    def get_last_layer(self):
        return getattr(self.decoder.decoder.colorize.module, 'weight_bar')

    def log_images(self, batch, split=""):
        log = dict()
        inputs = batch["image"].permute(0, 3, 1, 2)
        inputs = inputs.to(self.device)
        reconstructions, posterior = self(inputs)
        log["inputs"] = inputs
        log["reconstructions"] = reconstructions
        return log

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = batch["image"].permute(0, 3, 1, 2)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            output = pl.TrainResult(minimize=aeloss, checkpoint_on=aeloss)
            output.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return output

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            output = pl.TrainResult(minimize=discloss)
            output.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            output.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            del output["checkpoint_on"]      # NOTE pl currently sets checkpoint_on=minimize by default TODO
            return output

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"].permute(0, 3, 1, 2)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        output = pl.EvalResult(checkpoint_on=aeloss)
        output.log_dict(log_dict_ae)
        output.log_dict(log_dict_disc)
        return output

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters())+[self.loss.logvar],
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def on_epoch_end(self):
        pass


class BasicAE(pl.LightningModule):
    def __init__(self, ae_config, loss_config, ckpt_path=None, ignore_keys=[]):
        super().__init__()
        self.autoencoder = instantiate_from_config(ae_config)
        self.loss = instantiate_from_config(loss_config)
        if ckpt_path is not None:
            print("Loading model from {}".format(ckpt_path))
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        try:
            sd = torch.load(path, map_location="cpu")["state_dict"]
        except KeyError:
            sd = torch.load(path, map_location="cpu")

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)

    def forward(self, x):
        posterior = self.autoencoder.encode(x)
        h = posterior.sample()
        reconstructions = self.autoencoder.decode(h)
        return reconstructions, posterior

    def encode(self, x):
        posterior = self.autoencoder.encode(x)
        h = posterior.sample()
        return h

    def get_last_layer(self):
        return self.autoencoder.get_last_layer()

    def log_images(self, batch, split=""):
        log = dict()
        inputs = batch["image"].permute(0, 3, 1, 2)
        inputs = inputs.to(self.device)
        reconstructions, posterior = self(inputs)
        log["inputs"] = inputs
        log["reconstructions"] = reconstructions
        return log

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = batch["image"].permute(0, 3, 1, 2)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            output = pl.TrainResult(minimize=aeloss, checkpoint_on=aeloss)
            output.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return output

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            output = pl.TrainResult(minimize=discloss)
            output.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            output.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            del output["checkpoint_on"]      # NOTE pl currently sets checkpoint_on=minimize by default TODO
            return output

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"].permute(0, 3, 1, 2)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        output = pl.EvalResult(checkpoint_on=aeloss)
        output.log_dict(log_dict_ae)
        output.log_dict(log_dict_disc)
        return output

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.autoencoder.parameters())+[self.loss.logvar],
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
