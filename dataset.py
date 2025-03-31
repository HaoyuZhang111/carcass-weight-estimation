import argparse
import better_exceptions
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                ],
                random_order=True
        )

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img


class FaceDataset(Dataset):
    def __init__(self, data_dir, data_type, img_size=224, augment=True, age_stddev=1.0):
        assert(data_type in ("train_", "test_"))
        csv_path = Path(data_dir).joinpath(data_type).joinpath(f"data.csv")
        img_dir = Path(data_dir).joinpath(data_type)
        self.img_size = img_size
        self.augment = augment

        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = lambda i: i

        self.x_top = []
        self.x_right=[]
        self.y = []

        df = pd.read_csv(str(csv_path))

        for _, row in df.iterrows():
            img_name = row["id"]
            img_name = str(img_name).replace('.0', '').zfill(7)
            img_path=img_dir.joinpath(img_name + ".jpg")

            if img_path.is_file() == False:
                continue

            self.x_top.append(str(img_path))
            self.y.append(row["net_weight"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x_top[idx]
        weight = self.y[idx]

        img = cv2.imread(str(img_path), 1)
        img = self.transform(img)
        img_right = img[250:592, 0:699]
        img_top = img[0:250, 0:699]
        img_top = cv2.resize(img_top, (self.img_size, self.img_size)).astype(np.float32)
        img_right = cv2.resize(img_right, (self.img_size, self.img_size)).astype(np.float32)
        return torch.from_numpy(np.transpose(img_top, (2, 0, 1))), torch.from_numpy(
            np.transpose(img_right, (2, 0, 1))), np.clip(round(weight), 100, 600)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    dataset = FaceDataset(args.data_dir, "train")
    print("train dataset len: {}".format(len(dataset)))
    print(dataset)

if __name__ == '__main__':
    main()
