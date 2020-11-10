# Net2Net
Code accompanying the NeurIPS 2020 oral paper

[**Network-to-Network Translation with Conditional Invertible Neural Networks**](https://compvis.github.io/net2net/)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Patrick Esser](https://github.com/pesser)\*,
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
\* equal contribution

**tl;dr** Our approach distills the residual information of one model with respect to
another's and thereby enables translation between fixed off-the-shelve expert
models such as BERT and BigGAN without having to modify or finetune them.

![teaser](assets/teaser.png)
[BibTeX](#bibtex) | [Project Page](https://compvis.github.io/net2net/)

For now, example code for a translation between low-resolution and
high-resolution autoencoders is provided to obtain a generative superresolution
model. Other examples will be added until the [NeurIPS](https://nips.cc/) conference starts (December 2020).

## Requirements
A suitable [conda](https://conda.io/) environment named `net2net` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate net2net
```

## Datasets

- **CelebA-HQ**: Create a symlink `data/celebahq` pointing to a folder containing
  the `.npy` files of CelebA-HQ (instructions to obtain them can be found in
  the [PGGAN repository](https://github.com/tkarras/progressive_growing_of_gans)).
- **FFHQ**: Create a symlink `data/ffhq` pointing to the `images1024x1024` folder
  obtained from the [FFHQ repository](https://github.com/NVlabs/ffhq-dataset).


## Training
Our code uses [Pytorch-Lightning](https://www.pytorchlightning.ai/) and thus natively supports
things like 16-bit precision, multi-GPU training and gradient accumulation. Training details for any model need to be specified in a dedicated `.yaml` file.
In general, such a config file is structured as follows:
```
model:
  base_learning_rate: 4.5e-6
  target: <path/to/lightning/module>
  params:
    ...
data:
  target: translation.DataModuleFromConfig
  params:
    batch_size: ...
    num_workers: ...
    train:
      target: <path/to/train/dataset>
      params:
        ...
    validation:
      target: <path/to/validation/dataset>
      params:
        ...
```
Any Pytorch-Lightning model specified under `model.target` is then trained on the specified data
by running the command:
```
python translation.py --base <path/to/yaml> -t --gpus 0,
```
All available Pytorch-Lightning [trainer](https://pytorch-lightning.readthedocs.io/en/stable/trainer.html) arguments can be added via the command line, e.g. run
```
python translation.py --base <path/to/yaml> -t --gpus 0,1,2,3 --precision 16 --accumulate_grad_batches 2
```
to train a model on 4 GPUs using 16-bit precision and a 2-step gradient accumulation.
More details are provided in the examples below.

### Training a cINN
Training a cINN for network-to-network translation usually utilizes the Lighnting Module `net2net.models.flows.flow.Net2NetFlow`
and makes a few further assumptions on the configuration file and model interface:
```
model:
  base_learning_rate: 4.5e-6
  target: net2net.models.flows.flow.Net2NetFlow
  params:
    flow_config:
      target: <path/to/cinn>
      params:
        ...

    cond_stage_config:
      target: <path/to/network1>
      params:
        ...

    first_stage_config:
      target: <path/to/network2>
      params:
        ...
```
Here, the entries under `flow_config` specifies the architecture and parameters of the conditional INN; 
`cond_stage_config` specifies the first network whose representation is to be translated into another network
specified by `first_stage_config`.  Our model `net2net.models.flows.flow.Net2NetFlow` expects that the first  
network has a `.encode()` method which produces the representation of interest, while the second network should
have an `encode()` and a `decode()` method, such that both of them applied sequentially produce the networks output. This allows for a modular combination of arbitrary models of interest. For more details, see the examples below.

### Training a cINN - Examples
![superres](assets/superresolutionfigure.png) 
Training details for a cINN to concatenate two autoencoders from different image scales for stochastic
superresolution are specified in `configs/translation/faces32-to-256.yaml`. 

To train a model for translating from 32 x 32 images to 256 x 256 images on GPU 0, run
```
python translation.py --base configs/translation/faces32-to-faces256.yaml -t --gpus 0, 
``` 
and specify any additional training commands as described above. Note that this setup requires two
pretrained autoencoder models, one on 32 x 32 images and the other on 256 x 256. If you want to
train them yourself on a combination of FFHQ and CelebA-HQ, run
```
python translation.py --base configs/autoencoder/faces32 -t --gpus <n>, 
```
for the 32 x 32 images; and 
```
python translation.py --base configs/autoencoder/faces32 -t --gpus <n>, 
```
for the model on 256 x 256 images. After training, adopt the corresponding model paths in `configs/translation/faces32-to-faces256.yaml`. Additionally, we provide weights of pretrained autoencoders for both settings: 
[Weights 32x32](https://heibox.uni-heidelberg.de/f/b0b103af8406467abe48/);  [Weights256x256](https://heibox.uni-heidelberg.de/f/64f57a9dcbdc480f9178/). 
To run the training as described above, put them into 
`logs/2020-10-16T17-11-42_FacesFQ32x32/checkpoints/last.ckpt`and 
`logs/2020-09-16T16-23-39_FacesXL256z128/checkpoints/last.ckpt`, respectively.
## BibTeX

```
@misc{rombach2020net2net,
      title={Network-to-Network Translation with Conditional Invertible Neural Networks},
      author={Robin Rombach and Patrick Esser and Björn Ommer},
      year={2020},
}
```
