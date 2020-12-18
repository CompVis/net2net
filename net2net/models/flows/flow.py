import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from translation import instantiate_from_config
from net2net.modules.flow.loss import NLL
from net2net.ckpt_util import get_ckpt_path
from net2net.modules.util import log_txt_as_img


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Flow(pl.LightningModule):
    def __init__(self, flow_config):
        super().__init__()
        self.flow = instantiate_from_config(config=flow_config)
        self.loss = NLL()

    def forward(self, x):
        zz, logdet = self.flow(x)
        return zz, logdet

    def sample_like(self, query):
        z = self.flow.sample(query.shape[0], device=query.device).float()
        return z

    def shared_step(self, batch, batch_idx):
        x, labels = batch
        x = x.float()
        zz, logdet = self(x)
        loss, log_dict = self.loss(zz, logdet)
        return loss, log_dict

    def training_step(self, batch, batch_idx):
        loss, log_dict = self.shared_step(batch, batch_idx)
        output = pl.TrainResult(minimize=loss, checkpoint_on=loss)
        output.log_dict(log_dict, prog_bar=False, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.shared_step(batch, batch_idx)
        output = pl.EvalResult(checkpoint_on=loss)
        output.log_dict(log_dict, prog_bar=False)

        x, _ = batch
        x = x.float()
        sample = self.sample_like(x)
        output.sample_like = sample
        output.input = x.clone()

        return output

    def configure_optimizers(self):
        opt = torch.optim.Adam((self.flow.parameters()),lr=self.learning_rate, betas=(0.5, 0.9))
        return [opt], []


class Net2NetFlow(pl.LightningModule):
    def __init__(self,
                 flow_config,
                 first_stage_config,
                 cond_stage_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="image",
                 interpolate_cond_size=-1
                 ):
        super().__init__()
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        self.flow = instantiate_from_config(config=flow_config)
        self.loss = NLL()
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.interpolate_cond_size = interpolate_cond_size
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def init_cond_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.cond_stage_model = model

    def forward(self, x, c):
        c = self.encode_to_c(c)
        q = self.encode_to_z(x)
        zz, logdet = self.flow(q, c)
        return zz, logdet

    @torch.no_grad()
    def sample_conditional(self, c):
        z = self.flow.sample(c)
        return z

    @torch.no_grad()
    def encode_to_z(self, x):
        z = self.first_stage_model.encode(x).detach()
        return z

    @torch.no_grad()
    def encode_to_c(self, c):
        c = self.cond_stage_model.encode(c).detach()
        return c

    @torch.no_grad()
    def decode_to_img(self, z):
        x = self.first_stage_model.decode(z.detach())
        return x

    @torch.no_grad()
    def log_images(self, batch, split=""):
        log = dict()
        x = self.get_input(self.first_stage_key, batch).to(self.device)
        xc = self.get_input(self.cond_stage_key, batch, is_conditioning=True)
        if self.cond_stage_key not in ["text", "caption"]:
            xc = xc.to(self.device)

        z = self.encode_to_z(x)
        c = self.encode_to_c(xc)

        zz, _ = self.flow(z, c)
        zrec = self.flow.reverse(zz, c)
        xrec = self.decode_to_img(zrec)
        z_sample = self.sample_conditional(c)
        xsample = self.decode_to_img(z_sample)

        cshift = torch.cat((c[1:],c[:1]),dim=0)
        zshift = self.flow.reverse(zz, cshift)
        xshift = self.decode_to_img(zshift)

        log["inputs"] = x
        if self.cond_stage_key not in ["text", "caption", "class"]:
            log["conditioning"] = xc
        else:
            _,_,h,w = x.shape
            log["conditioning"] = log_txt_as_img((w,h), xc)

        log["reconstructions"] = xrec
        log["shift"] = xshift
        log["samples"] = xsample
        return log

    def get_input(self, key, batch, is_conditioning=False):
        x = batch[key]
        if key in ["caption", "text"]:
            x = list(x[0])
        elif key in ["class"]:
            pass
        else:
            if len(x.shape) == 3:
                x = x[..., None]
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            if is_conditioning:
                if self.interpolate_cond_size > -1:
                    x = F.interpolate(x, size=(self.interpolate_cond_size, self.interpolate_cond_size))
        return x

    def shared_step(self, batch, batch_idx, split="train"):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch, is_conditioning=True)
        zz, logdet = self(x, c)
        loss, log_dict = self.loss(zz, logdet, split=split)
        return loss, log_dict

    def training_step(self, batch, batch_idx):
        loss, log_dict = self.shared_step(batch, batch_idx, split="train")
        output = pl.TrainResult(minimize=loss, checkpoint_on=loss)
        output.log_dict(log_dict, prog_bar=False, on_epoch=True, logger=True, on_step=True)
        return output

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.shared_step(batch, batch_idx, split="val")
        output = pl.EvalResult(checkpoint_on=loss)
        output.log_dict(log_dict, prog_bar=False, logger=True)
        return output

    def configure_optimizers(self):
        opt = torch.optim.Adam((self.flow.parameters()),
                               lr=self.learning_rate,
                               betas=(0.5, 0.9),
                               amsgrad=True)
        return [opt], []


class Net2BigGANFlow(Net2NetFlow):
    def __init__(self,
                 flow_config,
                 gan_config,
                 cond_stage_config,
                 make_cond_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 cond_stage_key="caption"
                 ):
        super().__init__(flow_config=flow_config,
                         first_stage_config=gan_config, cond_stage_config=cond_stage_config,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, cond_stage_key=cond_stage_key
                         )

        self.init_to_c_model(make_cond_config)
        self.init_preprocessing()

    @torch.no_grad()
    def get_input(self, batch, move_to_device=False):
        zin = batch["z"]
        cin = batch["class"]
        if move_to_device:
            zin, cin = zin.to(self.device), cin.to(self.device)
        # dequantize the discrete class code
        cin = self.first_stage_model.embed_labels(cin, labels_are_one_hot=False)
        split_sizes = [zin.shape[1], cin.shape[1]]
        xin = self.first_stage_model.generate_from_embedding(zin, cin)
        cin = self.dequantizer(cin)
        xc = self.to_c_model(xin)
        zflow = torch.cat([zin, cin.detach()], dim=1)[:, :, None, None]  # this will be flowed
        return {"zcode": zflow,
                "xgen": xin,
                "xcon": xc,
                "split_sizes": split_sizes
                }

    def init_to_c_model(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.to_c_model = model

    def init_preprocessing(self):
        dqcfg = {"target": "net2net.modules.autoencoder.basic.BasicFullyConnectedVAE"}
        self.dequantizer = instantiate_from_config(dqcfg)
        ckpt = get_ckpt_path("dequant_vae", "net2net/modules/autoencoder/dequant_vae")
        self.dequantizer.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        self.dequantizer.eval()
        self.dequantizer.train = disabled_train

    def shared_step(self, batch, batch_idx, split="train"):
        data = self.get_input(batch)
        z, c = data["zcode"], data["xcon"]
        zz, logdet = self(z, c)
        loss, log_dict = self.loss(zz, logdet, split=split)
        return loss, log_dict

    def forward(self, z, c):
        c = self.encode_to_c(c)
        zz, logdet = self.flow(z, c)
        return zz, logdet

    @torch.no_grad()
    def log_images(self, batch, split=""):
        log = dict()
        data = self.get_input(batch, move_to_device=True)
        z, xc, x = data["zcode"], data["xcon"], data["xgen"]
        c = self.encode_to_c(xc)
        zz, _ = self.flow(z, c)

        z_sample = self.sample_conditional(c)
        zdec, cdec = torch.split(z_sample, data["split_sizes"], dim=1)
        xsample = self.first_stage_model.generate_from_embedding(zdec.squeeze(-1).squeeze(-1),
                                                                 cdec.squeeze(-1).squeeze(-1))

        cshift = torch.cat((c[1:],c[:1]),dim=0)
        zshift = self.flow.reverse(zz, cshift)
        zshift, cshift = torch.split(zshift, data["split_sizes"], dim=1)
        xshift = self.first_stage_model.generate_from_embedding(zshift.squeeze(-1).squeeze(-1),
                                                                cshift.squeeze(-1).squeeze(-1))

        log["inputs"] = x
        if self.cond_stage_key not in ["text", "caption", "class"]:
            log["conditioning"] = xc
        else:
            _,_,h,w = x.shape
            log["conditioning"] = log_txt_as_img((w,h), xc)

        log["shift"] = xshift
        log["samples"] = xsample
        return log

    @torch.no_grad()
    def sample_conditional(self, c):
        z = self.flow.sample(c)
        return z
