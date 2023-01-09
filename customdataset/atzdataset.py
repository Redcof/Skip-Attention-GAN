import ast
import os
import sys
from collections import defaultdict
import random

import PIL
import cv2
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import read_image


class ATZDataset(Dataset):
    CACHE_SIZE_BYTES = 1179200 * 2  # 10MB

    def __init__(self, atz_patch_dataset_csv, atz_dataset_train_or_test_txt, img_dir, transform=None,
                 label_transform=None, device="cpu", wavelet_transform=lambda x: x):
        # read trainable/testable files names for the experiment
        file_names = []
        with open(atz_dataset_train_or_test_txt, "r") as fp:
            file_names = [line.strip() for line in fp.readlines()]
        self.df = pd.read_csv(atz_patch_dataset_csv)
        if len(file_names) > 0:
            # filter dataframe
            self.df = self.df[self.df["image"].isin(file_names)]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = label_transform
        # each key: image name, value: transformed image data
        self.cache_image = defaultdict(lambda: None)  # process optimization
        self.one_image_size = 0  # bytes
        self.device = device
        self.wavelet_transform = wavelet_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Read a patch and return the item"""
        # read a record
        record = self.df.iloc[idx, :]
        # read metadata
        current_file = record[["image"]].values[0]
        label = record[["label"]].values[0]
        x1, x2, y1, y2 = ast.literal_eval(record[["x1x2y1y2"]].values[0])
        anomaly_size = record[["anomaly_size"]].values[0]

        # read image from cache
        pilimage = self.get_cached_image(current_file)
        image = pilimage.crop(box=(x1, y1, x2, y2))
        # image.show()
        # transform image
        if self.transform:
            image = self.transform(image)
        # cv2.imshow("patch", image)
        if self.target_transform:
            label = self.target_transform(label, anomaly_size)

        return image, label

    def cache_limit_check(self):
        """
        Adjust and randomly clear cache
        """
        items = len(self.cache_image.keys())
        total_bytes = items * self.one_image_size
        if total_bytes >= ATZDataset.CACHE_SIZE_BYTES:
            # cache full remove item
            self.cache_image.pop(random.choice(list(self.cache_image.keys())))

    def get_cached_image(self, current_file):
        """
            Read or update image cache. Returns an image object.
            return PIL image
        """
        # try to read image form cache
        image = self.cache_image[current_file]
        if image is None:
            # prepare image path
            img_path = os.path.join(self.img_dir, current_file)
            # read imagedata
            image = Image.open(img_path)
            # convert to greyscale
            # image = image.convert("L")
            image = self.wavelet_transform(image)
            if self.one_image_size == 0:
                self.one_image_size = 1179200
            # check cache size
            self.cache_limit_check()
            # save image to cache
            self.cache_image[current_file] = image
        return image