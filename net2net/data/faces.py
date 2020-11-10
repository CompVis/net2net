import os
from torch.utils.data import Dataset, ConcatDataset
from net2net.data.base import ImagePaths, NumpyPaths


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
    def __init__(self, size):
        super().__init__()
        root = "data/celebahq"
        with open("data/celebahqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)


class CelebAHQValidation(FacesBase):
    def __init__(self, size):
        super().__init__()
        root = "data/celebahq"
        with open("data/celebahqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)


class FFHQTrain(FacesBase):
    def __init__(self, size):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class FFHQValidation(FacesBase):
    def __init__(self, size):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CelebFQTrain(Dataset):
    def __init__(self, size):
        d1 = CelebAHQTrain(size=size)
        d2 = FFHQTrain(size=size)
        self.data = ConcatDataset([d1, d2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class CelebFQValidation(Dataset):
    def __init__(self, size):
        d1 = CelebAHQValidation(size=size)
        d2 = FFHQValidation(size=size)
        self.data = ConcatDataset([d1, d2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


if __name__ == "__main__":
    from pprint import pprint

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
