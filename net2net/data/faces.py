import os, pickle
import albumentations
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from net2net.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
import net2net.data.utils as ndu


class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex


class CelebAHQTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/celebahq"
        with open("data/celebahqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class CelebAHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/celebahq"
        with open("data/celebahqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FFHQTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FFHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class CelebABase(Dataset):
    NAME = "CelebA"
    URL = "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
    FILES = [
        "img_align_celeba.zip",
        "list_eval_partition.txt",
        "identity_CelebA.txt",
        "list_attr_celeba.txt",
    ]

    def __init__(self, config=None):
        self.config = config or dict()
        self._prepare()
        self._load()

    def _prepare(self):
        self.root = ndu.get_root(self.NAME)
        self._data_path = Path(self.root).joinpath("data.p")
        if not ndu.is_prepared(self.root):
            print("preparing CelebA dataset...")
            # prep
            root = Path(self.root)
            local_files = dict()

            local_files[self.FILES[0]] = ndu.prompt_download(
                self.FILES[0], self.URL, root, content_dir="img_align_celeba"
            )
            if not os.path.exists(os.path.join(root, "img_align_celeba")):
                print("Extracting {}".format(local_files[self.FILES[0]]))
                ndu.unpack(local_files["img_align_celeba.zip"])

            for v in self.FILES[1:]:
                local_files[v] = ndu.prompt_download(v, self.URL, root)

            with open(os.path.join(self.root, "list_eval_partition.txt"), "r") as f:
                list_eval_partition = f.read().splitlines()
                fnames = [s[:10] for s in list_eval_partition]
                list_eval_partition = np.array(
                    [int(s[11:]) for s in list_eval_partition]
                )
            with open(os.path.join(self.root, "list_attr_celeba.txt"), "r") as f:
                list_attr_celeba = f.read().splitlines()
                attribute_descriptions = list_attr_celeba[1]
                list_attr_celeba = list_attr_celeba[2:]
                assert len(list_attr_celeba) == len(list_eval_partition)
                assert [s[:10] for s in list_attr_celeba] == fnames
                list_attr_celeba = np.array(
                    [[int(x) for x in s[11:].split()] for s in list_attr_celeba]
                )
            with open(os.path.join(self.root, "identity_CelebA.txt"), "r") as f:
                identity_celeba = f.read().splitlines()
                assert [s[:10] for s in identity_celeba] == fnames
                identity_celeba = np.array([int(s[11:]) for s in identity_celeba])

            data = {
                "fname": np.array(
                    [os.path.join("img_align_celeba/{}".format(s)) for s in fnames]
                ),
                "partition": list_eval_partition,
                "identity": identity_celeba,
                "attributes": list_attr_celeba,
            }
            with open(self._data_path, "wb") as f:
                pickle.dump(data, f)
            ndu.mark_prepared(self.root)

    def _get_split(self):
        split = (
            "test" if self.config.get("test_mode", False) else "train"
        )  # default split
        if self.NAME in self.config:
            split = self.config[self.NAME].get("split", split)
        return split

    def _load(self):
        with open(self._data_path, "rb") as f:
            self._data = pickle.load(f)
        split = self._get_split()
        assert split in ["train", "test", "val"]
        print("Using split: {}".format(split))
        if split == "train":
            self.split_indices = np.where(self._data["partition"] == 0)[0]
        elif split == "val":
            self.split_indices = np.where(self._data["partition"] == 1)[0]
        elif split == "test":
            self.split_indices = np.where(self._data["partition"] == 2)[0]
        self.labels = {
            "fname": self._data["fname"][self.split_indices],
            "partition": self._data["partition"][self.split_indices],
            "identity": self._data["identity"][self.split_indices],
            "attributes": self._data["attributes"][self.split_indices],
        }
        self._length = self.labels["fname"].shape[0]

    def _load_example(self, i):
        example = dict()
        for k in self.labels:
            example[k] = self.labels[k][i]
        example["image"] = Image.open(os.path.join(self.root, example["fname"]))
        if not example["image"].mode == "RGB":
            example["image"] = example["image"].convert("RGB")
        example["image"] = np.array(example["image"])
        return example

    def _preprocess_example(self, example):
        example["image"] = example["image"] / 127.5 - 1.0
        example["image"] = example["image"].astype(np.float32)

    def __getitem__(self, i):
        example = self._load_example(i)
        self._preprocess_example(example)
        return example

    def __len__(self):
        return self._length


class CelebA(CelebABase):
    """CelebA with support for resizing and fixed cropping as in lucic2018"""
    def __init__(self, config):
        super().__init__(config)
        self.size = config["spatial_size"]
        self.attribute_descriptions = [
            "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive",
            "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
            "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
            "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses",
            "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
            "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes",
            "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
            "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling",
            "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
            "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
            "Wearing_Necktie", "Young"]
        self.cropper = albumentations.CenterCrop(height=160,width=160)
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.preprocessor = albumentations.Compose([self.cropper, self.rescaler])

        if "cropsize" in config and config["cropsize"] < self.size:
            self.cropsize = config["cropsize"]
            self.preprocessor = albumentations.Compose([
                self.preprocessor,
                albumentations.RandomCrop(height=self.cropsize, width=self.cropsize)])

    def _preprocess_example(self, example):
        example["image"] = self.preprocessor(image=example["image"])["image"]
        example["image"] = (example["image"] + np.random.random()) / 256.  # dequantization
        example["image"] = (255 * example["image"])
        return super()._preprocess_example(example)

    def __getitem__(self, i):
        attr = self.labels['attributes'][i]
        example = super().__getitem__(i)
        example['attribute'] = attr
        example['index'] = i
        return example


class _CelebATrain(CelebA):
    def _get_split(self):
        return "train"


class _CelebATest(CelebA):
    def _get_split(self):
        return "test"


class CelebATrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        cfg = {"spatial_size": size}
        self.data = _CelebATrain(cfg)
        self.keys = keys


class CelebAValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        cfg = {"spatial_size": size}
        self.data = _CelebATest(cfg)
        self.keys = keys


class CelebFQTrain(Dataset):
    def __init__(self, size):
        d1 = CelebAHQTrain(size=size)
        d2 = FFHQTrain(size=size)
        self.data = ConcatDatasetWithIndex([d1, d2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example, y = self.data[i]
        example["class"] = y
        return example


class CelebFQValidation(Dataset):
    def __init__(self, size):
        d1 = CelebAHQValidation(size=size)
        d2 = FFHQValidation(size=size)
        self.data = ConcatDatasetWithIndex([d1, d2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example, y = self.data[i]
        example["class"] = y
        return example


class CCFQTrain(Dataset):
    """CelebA, CelebA-HQ and FFHQ"""
    def __init__(self, size):
        d1 = CelebATrain(size=size, keys=["image"])
        d2 = CelebAHQTrain(size=size, keys=["image"])
        d3 = FFHQTrain(size=size, keys=["image"])
        self.data = ConcatDatasetWithIndex([d1, d2, d3])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example, y = self.data[i]
        example["class"] = y
        return example


class CCFQValidation(Dataset):
    """CelebA, CelebA-HQ and FFHQ"""
    def __init__(self, size):
        d1 = CelebAValidation(size=size, keys=["image"])
        d2 = CelebAHQValidation(size=size, keys=["image"])
        d3 = FFHQValidation(size=size, keys=["image"])
        self.data = ConcatDatasetWithIndex([d1, d2, d3])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example, y = self.data[i]
        example["class"] = y
        return example


class AnimeFacesTrain(FacesBase):
    """Anime Faces obtained from Gwern's https://www.gwern.net/Crops """
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/anime"
        with open("data/animegwerncroptrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class AnimeFacesValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/anime"
        with open("data/animegwerncropvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FacesHQAndAnimeTrain(Dataset):
    # (FFHQ+CeleA-HQ) [0] + Anime [1]
    def __init__(self, size):
        super().__init__()
        d1 = ConcatDataset([FFHQTrain(size=size, keys=["image"]), CelebAHQTrain(size=size, keys=["image"])])
        d2 = AnimeFacesTrain(size=size, keys=["image"])
        self.data = ConcatDatasetWithIndex([d1, d2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        ex["class"] = y
        return ex


class FacesHQAndAnimeValidation(Dataset):
    # (FFHQ+CeleA-HQ) [0] + Anime [1]
    def __init__(self, size):
        super().__init__()
        d1 = ConcatDataset([FFHQValidation(size=size, keys=["image"]),
                            CelebAHQValidation(size=size, keys=["image"])])
        d2 = AnimeFacesValidation(size=size, keys=["image"])
        self.data = ConcatDatasetWithIndex([d1, d2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        ex["class"] = y
        return ex


class OilPortraitsTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/portraits"
        with open("data/portraitstrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class OilPortraitsValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/portraits"
        with open("data/portraitsvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FacesHQAndPortraitsTrain(Dataset):
    # (FFHQ+CeleA-HQ) [0] + Portraits [1]
    def __init__(self, size):
        super().__init__()
        d1 = ConcatDataset([FFHQTrain(size=size, keys=["image"]), CelebAHQTrain(size=size, keys=["image"])])
        d2 = OilPortraitsTrain(size=size, keys=["image"])
        self.data = ConcatDatasetWithIndex([d1, d2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        ex["class"] = y
        return ex


class FacesHQAndPortraitsValidation(Dataset):
    # (FFHQ+CeleA-HQ) [0] + Portraits [1]
    def __init__(self, size):
        super().__init__()
        d1 = ConcatDataset([FFHQValidation(size=size, keys=["image"]),
                            CelebAHQValidation(size=size, keys=["image"])])
        d2 = OilPortraitsValidation(size=size, keys=["image"])
        self.data = ConcatDatasetWithIndex([d1, d2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        ex["class"] = y
        return ex


class FFHQAndPortraitsTrain(Dataset):
    # FFHQ [0] + Portraits [1]
    def __init__(self, size):
        super().__init__()
        d1 = FFHQTrain(size=size, keys=["image"])
        d2 = OilPortraitsTrain(size=size, keys=["image"])
        self.data = ConcatDatasetWithIndex([d1, d2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        ex["class"] = y
        return ex


class FFHQAndPortraitsValidation(Dataset):
    # FFHQ [0] + Portraits [1]
    def __init__(self, size):
        super().__init__()
        d1 = FFHQValidation(size=size, keys=["image"])
        d2 = OilPortraitsValidation(size=size, keys=["image"])
        self.data = ConcatDatasetWithIndex([d1, d2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        ex["class"] = y
        return ex

if __name__ == "__main__":

    d = FFHQTrain(size=256)
    print("size FFHQTrain:", len(d))
    d = FFHQValidation(size=256)
    print("size FFHQValidation:", len(d))
    x = d[0]["image"]
    print(x.shape)
    print(type(x))
    print(x.max(), x.min())

    d = CelebAHQTrain(size=256)
    print("size CelebAHQTrain:", len(d))
    d = CelebAHQValidation(size=256)
    print("size CelebAHQValidation:", len(d))
    x = d[0]["image"]
    print(x.shape)
    print(type(x))
    print(x.max(), x.min())

    d = CelebATrain(size=256)
    print("size CelebATrain:", len(d))
    d = CelebAValidation(size=256)
    print("size CelebAValidation:", len(d))
    x = d[0]["image"]
    print(x.shape)
    print(type(x))
    print(x.max(), x.min())

    d = CCFQTrain(size=256)
    print("size CCFQTrain:", len(d))
    d = CCFQValidation(size=256)
    print("size CCFQ:", len(d))
    x = d[0]["image"]
    print(x.shape)
    print(type(x))
    print(x.max(), x.min())