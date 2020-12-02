import os, math
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset


def get_preprocessor(size=None, random_crop=False, additional_targets=None):
    if size is not None and size > 0:
        transforms = list()
        rescaler = albumentations.SmallestMaxSize(max_size = size)
        transforms.append(rescaler)
        if not random_crop:
            cropper = albumentations.CenterCrop(height=size,width=size)
            transforms.append(cropper)
        else:
            cropper = albumentations.RandomCrop(height=size,width=size)
            transforms.append(cropper)
            flipper = albumentations.HorizontalFlip()
            transforms.append(flipper)
        preprocessor = albumentations.Compose(transforms,
                                              additional_targets=additional_targets)
    else:
        preprocessor = lambda **kwargs: kwargs
    return preprocessor


# deepfashion with openpose keypoints as in CoCosNet

class DeepFashionBase(Dataset):
    # folder containing https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E
    INSHOP_ROOT = "data/deepfashion_inshop"
    # folder containing unzipped pose.zip from
    # https://drive.google.com/file/d/1Vzpl3DpHZistiEjXXb0Blk4L12LsDluU/view
    # and train.txt, val.txt from
    # https://drive.google.com/drive/folders/1kLOeRYZ1wUDzo3eg9ZihJj-yuyDQhp_T
    POSE_ROOT = "data/deepfashion_cocosnet_pose"

    def __init__(self, config=None, size=256, random_crop=False):
        self.preprocessor = get_preprocessor(
            size, random_crop, additional_targets={"pose": "image"})
        self.split = self.get_split()
        self.data_csv = {"train": os.path.join(self.POSE_ROOT, "train.txt"),
                         "validation": os.path.join(self.POSE_ROOT, "val.txt")}[self.split]
        with open(self.data_csv, "r") as f:
            self.image_paths = [l.replace("\\", "/") for l in f.read().splitlines()]
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": self.image_paths,
            "file_path_": [os.path.join(self.INSHOP_ROOT, "Img", l)
                           for l in self.image_paths],
            "pose_candidate_path_": [os.path.join(
                self.POSE_ROOT,
                l.replace("img", "pose", 1).replace('.jpg', '_candidate.txt'))
                                     for l in self.image_paths],
            "pose_subset_path_": [os.path.join(
                self.POSE_ROOT,
                l.replace("img", "pose", 1).replace('.jpg', '_subset.txt'))
                                     for l in self.image_paths]
        }

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        candidate = np.loadtxt(example["pose_candidate_path_"])
        subset = np.loadtxt(example["pose_subset_path_"])

        stickwidth = 4
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                   [1, 16], [16, 18], [3, 17], [6, 18]]

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], \
                  [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], \
                  [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

        img_path = example["file_path_"]
        img = cv2.imread(img_path)
        canvas = np.zeros_like(img)
        for i in range(18):
            index = int(subset[i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
        joints = []
        for i in range(17):
            index = subset[np.array(limbSeq[i]) - 1]
            cur_canvas = canvas.copy()
            if -1 in index:
                joints.append(np.zeros_like(cur_canvas[:, :, 0]))
                continue
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                       (int(length / 2), stickwidth),
                                       int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

            joint = np.zeros_like(cur_canvas[:, :, 0])
            cv2.fillConvexPoly(joint, polygon, 255)
            joint = cv2.addWeighted(joint, 0.4, joint, 0.6, 0)
            joints.append(joint)

        pose = np.array(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
        tensors_dist = list()
        for i in range(len(joints)):
            im_dist = cv2.distanceTransform(255-joints[i], cv2.DIST_L1, 3)
            im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
            tensor_dist = np.array(Image.fromarray(im_dist))
            tensors_dist.append(tensor_dist)

        tensors_dist = np.stack(tensors_dist, -1)

        tensor_pose = pose
        label_tensor = np.concatenate((tensor_pose, tensors_dist), axis=-1)

        img = np.array(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        example["image"] = img
        example["pose"] = label_tensor

        for k in ["image", "pose"]:
            example[k] = (example[k]/127.5-1.0).astype(np.float32)

        transformed = self.preprocessor(image=example["image"],
                                        pose=example["pose"])
        for k in ["image", "pose"]:
            example[k] = transformed[k]

        example["stickman"] = example["pose"][:,:,:3]

        return example


class DeepFashionTrain(DeepFashionBase):
    def __init__(self, config=None, size=256, random_crop=True):
        super().__init__(config, size, random_crop)

    def get_split(self):
        return "train"


class DeepFashionValidation(DeepFashionBase):
    def get_split(self):
        return "validation"
