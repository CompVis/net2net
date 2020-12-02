import argparse, os, sys, glob
import io
import albumentations
import torch
import numpy as np
from omegaconf import OmegaConf
import streamlit as st
import scipy
from scipy import ndimage
from streamlit import caching
from PIL import Image
from torchvision.utils import make_grid
from translation import instantiate_from_config, DataModuleFromConfig

DLIB_MSG = ("Please install `dlib`, download "+
            "`http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2` "+
            "and extract to `data/shape_predictor_68_face_landmarks.dat` in " +
            "order to align and crop custom face images.")
try:
    import dlib
except ModuleNotFoundError:
    dlib = None
    print(DLIB_MSG)

st.set_option('deprecation.showfileUploaderEncoding', False)
rescale = lambda x: (x + 1.) / 2.

DEBUG = False
DLIB_ROOT = "data"
CKPT_ROOT = "logs"

def predict_landmarks(img, predictor_path=os.path.join(
        DLIB_ROOT, "shape_predictor_68_face_landmarks.dat")):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)

    num_faces = len(dets)
    assert num_faces > 0, "Sorry, there were no faces found! Try with disabled 'Human Face Detection'."
    if DEBUG:
        print("Number of faces detected: {}".format(num_faces))

    for k, d in enumerate(dets):
        if DEBUG:
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        landmarks = list()
        for p in shape.parts():
            landmarks.append(np.array([p.x, p.y]))
        if DEBUG:
            print("{} landmarks in total".format(len(shape.parts())))

    return np.array(landmarks) # only return last detected shape


def make_aligned_images(img, face_landmarks, dst_dir='realign1024x1024', output_size=256,
                        transform_size=1024, enable_padding=True):
    """function modfied from https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py"""

    # Parse landmarks.
    # pylint: disable=unused-variable
    lm = face_landmarks
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.shape[0]) / shrink)), int(np.rint(float(img.shape[1]) / shrink)))
        img = Image.fromarray(img)
        img = img.resize(rsize, Image.ANTIALIAS)
        img = np.array(img)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.shape[0]), min(crop[3] + border, img.shape[1]))
    if crop[2] - crop[0] < img.shape[0] or crop[3] - crop[1] < img.shape[1]:
        img = Image.fromarray(img)
        img = img.crop(crop)
        img = np.array(img)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.shape[0] + border, 0), max(pad[3] - img.shape[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # transform.
    try:
        img = Image.fromarray(img)
    except:
        pass
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    img = np.array(img)
    # all done.
    if DEBUG:
        print("cropped to content.")
    return img


def crop_face(image, h=160, w=160):
    cropper = albumentations.CenterCrop(height=h, width=w)
    rescaler = albumentations.SmallestMaxSize(max_size=256)
    preprocessor = albumentations.Compose([cropper, rescaler])
    image = preprocessor(image=image)["image"]
    return image

def rescale_face(image):
    rescaler = albumentations.SmallestMaxSize(max_size=256)
    preprocessor = albumentations.Compose([rescaler])
    image = preprocessor(image=image)["image"]
    return image


def get_interactive_image():
    image = st.sidebar.file_uploader("Upload Custom Image", type=["jpg", "JPEG", "png"])
    if image is not None:
        has_dlib = dlib is not None
        if not has_dlib:
            st.info(DLIB_MSG)
        image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if st.sidebar.checkbox("Use Human Face Detection Algorithm", value=has_dlib):
            lm = predict_landmarks(image)
            image = make_aligned_images(image, lm)
        else:
            st.info("Warning: Model might fail if detection is disabled.")
            img = Image.fromarray(image)
            img = img.resize((256, 256))
            image = np.array(img)
        return image


def bchw_to_st(x, grid=False, nrow=4):
    if grid:
        x = x.detach().cpu()
        grid = make_grid(x, pad_value=1, padding=8, nrow=nrow)
        grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.numpy()
        grid = (grid * 255).astype(np.uint8)
        return grid
    return rescale(x.detach().cpu().numpy().transpose(0,2,3,1))


def single_image_to_torch(x):
    assert x is not None, "Please provide an image through the upload function"
    x = np.array(x)
    x = torch.FloatTensor(x/255.*2. - 1.)[None,...]
    return x


def build_random_batch(dset, batch_size, idx=None):
    keys = dset[0].keys()
    batch = {k:list() for k in keys}
    for b in range(batch_size):
        ridx = np.random.randint(0, len(dset)) if idx is None else idx
        ex = dset[ridx]
        for k in keys:
            batch[k].append(torch.tensor(ex[k]))
    for k in keys:
        batch[k] = torch.stack(batch[k])
    return batch


def give_other_labels(label, label_dict):
    other_labels = [l for l in label_dict if l is not label.item()]
    return other_labels


@torch.no_grad()
def sample_all_labels(model, num_samples, label_dict):
    zz_sample = torch.randn(num_samples, 128, 1, 1).to(model.device)
    decoded_samples = list()
    for label in label_dict:
        cond = torch.LongTensor([label])
        cond = model.cond_stage_model.make_one_hot(cond).to(model.device)
        cond = cond.repeat([num_samples, 1])
        z_sample = model.flow.reverse(zz_sample, cond)
        decoded_samples.append(model.decode_to_img(z_sample))
    return decoded_samples


@torch.no_grad()
def modify_input(model, example, label_dict, alpha=1.0):
    log = dict()
    inputs = example["image"].permute(0, 3, 1, 2)
    label = example["class"]
    inputs = inputs.to(model.device)
    cond = model.cond_stage_model.make_one_hot(label).to(inputs)
    z = model.first_stage_model.encode(inputs, return_mode=True)
    zz, _ = model.flow(z, cond)

    x_inv_rec = list()
    other_labels = give_other_labels(label, label_dict)
    for other_label in other_labels:
        cond = torch.LongTensor([other_label])
        cond = model.cond_stage_model.make_one_hot(cond).to(model.device)
        z_inv_rec = model.flow.reverse(zz, cond)
        if alpha != 1.0:
            z_inv_rec = (1.0-alpha)*z+alpha*z_inv_rec
        x_inv_rec.append(model.decode_to_img(z_inv_rec))

    log["inputs"] = inputs
    log["modified"] = x_inv_rec
    return log, label.item(), other_labels


def run(model, dset, label_dict, mode):
    rev_ld = dict((v,k) for k, v in label_dict.items())
    st.sidebar.subheader("Input")
    image = get_interactive_image()
    if image is not None:
        image = single_image_to_torch(image)

        in_label = st.sidebar.selectbox("Input Label", list(label_dict.values())[::-1])
        in_label = rev_ld[in_label]
        example = {"image":image, "class": torch.LongTensor([in_label])}
    else:
        idx = st.sidebar.slider("Or Use Example from Dataset", 0, len(dset) - 1, value=0)
        example = dset[idx]
        in_label = label_dict[example["class"]]
        st.sidebar.write("Input Label: {}".format(in_label))
        example["image"] = torch.tensor(example["image"])[None, ...]
        example["class"] = torch.LongTensor([example["class"]])

    # KISS
    #st.sidebar.subheader("Interpolate Translation")
    #alpha = st.sidebar.slider("modification strength", value=1.0, min_value=0.0,
    #                          max_value=1.0, step=0.05)
    alpha = 1.0

    other_labels = give_other_labels(example["class"], label_dict)
    other_keys = [label_dict[l] for l in other_labels]
    st.sidebar.subheader("Animate Translation")
    animation_target = st.sidebar.selectbox("Animation Target Label",
                                            other_keys)
    animation_target_idx = other_keys.index(animation_target)
    animate = st.sidebar.button("animate translation")

    # go
    st.header("{} - Input Modification".format(mode))
    output_text = st.empty()
    output_images = st.empty()
    info = st.empty()

    if animate:
        if "CelebA" in [animation_target, in_label]:
            st.sidebar.info("Animated translations involving CelebA suffer from "+
                            "ghosting due to differences in alignment of the datasets.")
        import imageio
        outvid = "interpolation.mp4"
        writer = imageio.get_writer(outvid, fps=25)
        for alpha in np.linspace(0.0, 1.0, 5*25):
            logs, current_label, other_labels = modify_input(model, example,
                                                             label_dict=label_dict,
                                                             alpha=alpha)
            x = logs["modified"][animation_target_idx]
            x = x.cpu().numpy().transpose(0,2,3,1)
            x = ((x+1.0)*127.5).astype(np.uint8)
            writer.append_data(x[0])
            output_images.image(x)
            info.write("{}%".format(int(100*alpha)))
        writer.close()
        st.video(outvid)
    else:
        logs, current_label, other_labels = modify_input(model, example,
                                                         label_dict=label_dict,
                                                         alpha=alpha)
    images = torch.cat([logs["inputs"], torch.cat(logs["modified"])])

    modified_labels_string = "("
    for k in other_labels:
        modified_labels_string += label_dict[k]+", "
    modified_labels_string = modified_labels_string[:-2]+")"

    output_text.write(
        "Input **({})** & Modified Output **{}**".format(label_dict[current_label], modified_labels_string))
    output_images.image(bchw_to_st(images, grid=True, nrow=len(label_dict.keys())), clamp=True)

    st.write("________________________________________________")
    # samples
    st.header("{} - Samples".format(mode))
    labels_as_string = ""
    for k in label_dict:
        labels_as_string += label_dict[k]+", "
    st.write("Generations of order: **{}**".format(labels_as_string[:-2]))

    st.sidebar.subheader("Sampling")
    num_samples = st.sidebar.slider("Number of Samples", 1, 4, value=2)
    st.sidebar.button("Sample again")
    samples = sample_all_labels(model, num_samples, label_dict)
    samples = torch.stack(samples)
    for n in range(num_samples):
        st.image(bchw_to_st(samples[:,n,...], grid=True, nrow=len(label_dict.keys())), clamp=True)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    return parser


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    model = instantiate_from_config(config)
    if sd is not None:
        model.load_state_dict(sd)
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


def get_data(config):
    # get data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    dset = data.datasets["validation"]
    return dset


@st.cache(allow_output_mutation=True)
def load_model_and_dset(config, ckpt, gpu, eval_mode):
    # get data
    dset = get_data(config)   # calls data.config ...

    # now load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"],
                                   gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    return dset, model, global_step


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    mode = st.sidebar.selectbox("Model Edition", ["CelebA - CelebaHQ - FFHQ",
                                                  "Anime - Photography",
                                                  "Portrait - Photography",
    ])

    if mode == "CelebA - CelebaHQ - FFHQ":
        path = os.path.join(CKPT_ROOT, "2020-11-30T23-32-28_celeba_celebahq_ffhq_256")
        ckpt = os.path.join(CKPT_ROOT, "2020-11-30T23-32-28_celeba_celebahq_ffhq_256/checkpoints/last.ckpt")
        label_dict = {0: "CelebA", 1: "CelebaHQ", 2: "FFHQ"}
    elif mode == "Anime - Photography":
        path = os.path.join(CKPT_ROOT, "2020-12-02T13-58-19_anime_photography_256")
        ckpt = os.path.join(CKPT_ROOT, "2020-12-02T13-58-19_anime_photography_256/checkpoints/epoch=000004.ckpt")
        label_dict = {0: "Photography", 1: "Anime"}
        st.sidebar.info("Note that the anime dataset contains mostly female characters.")
    elif mode == "Portrait - Photography":
        path = os.path.join(CKPT_ROOT, "2020-12-02T16-19-39_portraits_photography_256")
        ckpt = os.path.join(CKPT_ROOT, "2020-12-02T16-19-39_portraits_photography_256/checkpoints/epoch=000003.ckpt")
        label_dict = {0: "Photography", 1: "Oil Portrait"}
    else:
        raise ValueError("Unknown mode {}".format(mode))

    logdir = path.rstrip("/")
    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))

    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    config = OmegaConf.merge(*configs)
    try:
        # first stage is contained in checkpoint; no need to load twice
        del config.model.params.first_stage_config.params["ckpt_path"]
    except:
        pass
    if DEBUG:
        print(config)

    gpu = torch.cuda.is_available()
    eval_mode = True
    show_config = False
    # KISS
    #if st.sidebar.checkbox("More Options"):
    #    gpu = st.sidebar.checkbox("GPU", value=gpu)
    #    eval_mode = st.sidebar.checkbox("Eval Mode", value=eval_mode)
    #    show_config = st.sidebar.checkbox("Show Config", value=show_config)
    #    if show_config:
    #        st.info("Checkpoint: {}".format(ckpt))
    #        st.json(OmegaConf.to_container(config))

    dset, model, global_step = load_model_and_dset(config, ckpt, gpu, eval_mode)
    if DEBUG:
        st.info("Global Step: {}".format(global_step))

    run(model, dset, label_dict=label_dict, mode=mode)
