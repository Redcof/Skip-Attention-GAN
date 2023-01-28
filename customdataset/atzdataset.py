import ast
import os
from collections import defaultdict
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from empatches import EMPatches
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ATZDataset(Dataset):
    CACHE_ITEM_LIMIT = 5  # 5 files
    NORMAL = 0
    ABNORMAL = 1

    def __init__(self, atz_patch_dataset_csv, img_dir, phase, atz_dataset_train_or_test_txt=None, transform=None,
                 classes=(), subjects=(), ablation=0, label_transform=None, device="cpu",
                 patch_size=128, patch_overlap=0.2, balanced=False,
                 train_split=None, test_split=None, nc=3,
                 global_wavelet_transform=lambda x: x, random_state=47):
        assert train_split is None or test_split is None, ("Either of train_split and test_split is required."
                                                           "But both values "
                                                           "train_split=%f and test_split=%f are provided." % (
                                                               train_split, test_split))
        assert phase in ["train", "test"], "Invalid phase value='%s'" % phase
        if train_split is None and test_split is None:
            self.train_split = 0.8
        elif test_split is not None:
            self.train_split = 1 - test_split
        if subjects is None:
            subjects = []
        # label transformation function
        if label_transform is None:
            label_transform = self.label_transform_default
        self.label_transform = label_transform
        self.phase = phase
        self.img_dir = img_dir
        self.transform = transform
        # each key: image name, value: transformed image data
        self.cache_image = defaultdict(lambda: None)  # process optimization
        self.device = device
        self.gbl_wavelet_transform = global_wavelet_transform
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.random_state = random_state
        self.balanced = balanced
        self.nc = nc
        self.normal_count = 0
        self.abnormal_count = 0
        # read CSV
        self.df = pd.read_csv(atz_patch_dataset_csv)
        self.filter(atz_dataset_train_or_test_txt, classes, subjects)
        self.ablation = ablation

        def _lambda(record, **kwargs):
            return self.label_transform(record.image, record.label_txt, record.anomaly_size)

        # perform label-transform
        self.df['is_anamoly'] = self.df.apply(_lambda, axis=1)

        self.split()
        self.shuffle()

    def shuffle(self):
        # shuffle dataframe
        self.df = self.df.sample(frac=1, random_state=self.random_state)
        # shuffle dataframe again
        self.df = self.df.sample(frac=1, random_state=self.random_state + 1)

    def filter(self, atz_dataset_train_or_test_txt, classes, subjects):
        # read trainable/testable files names for the experiment
        file_names = []
        if atz_dataset_train_or_test_txt is not None and os.path.exists(atz_dataset_train_or_test_txt):
            print(atz_dataset_train_or_test_txt)
            with open(atz_dataset_train_or_test_txt, "r") as fp:
                file_names = [line.strip() for line in fp.readlines()]
        if file_names:
            # filter dataframe
            self.df = self.df[self.df["image"].isin(file_names)]
        if classes:
            if self.phase == "test":
                self.df = self.df[self.df['label_txt'].isin(classes)]
            elif self.phase == "train":
                self.df = self.df[self.df['label_txt'].isin(["NORMAL0", "NORMAL1"])]
        if subjects:
            self.df = self.df[self.df['subject_id'].isin(subjects)]

    def split(self):
        df_abnormal = self.df[self.df["is_anamoly"] == 1]
        df_normal = self.df[self.df["is_anamoly"] == 0]
        if self.ablation:
            test_items = self.ablation * self.train_split
            train_items = self.ablation / (1 - self.train_split)
            items = int(test_items + train_items)
            # select total normal samples for train and test
            if len(df_normal) > items:
                df_normal = df_normal.sample(items, random_state=self.random_state)
            if self.phase == "test":
                # select abnormal samples for testing
                df_abnormal = df_abnormal.sample(self.ablation, random_state=self.random_state)

        abnormal_count = len(df_abnormal)
        norm_len = len(df_normal)
        # split dataframe into train and test
        split_len = int(norm_len * self.train_split)
        if self.phase == "train":
            df_normal = df_normal.iloc[:split_len, :]
            norm_len = len(df_normal)
        else:
            df_normal = df_normal.iloc[split_len:, :]
            norm_len = len(df_normal)

        if self.balanced and self.phase == "test":
            if norm_len > abnormal_count:
                df_normal = df_normal.sample(abnormal_count, random_state=self.random_state)

        # concat normal and abnormal data
        if self.phase == "test":
            self.df = pd.concat([df_abnormal, df_normal])
            norm_len = len(df_normal)
            abnormal_count = len(df_abnormal)
        elif self.phase == "train":
            self.df = pd.concat([df_normal])
            norm_len = len(df_normal)
            abnormal_count = 0

        # recalculate
        self.normal_count = norm_len
        self.abnormal_count = abnormal_count
        # msg = "Phase %s => Normal:Abnormal = %d:%d" % (self.phase, self.normal_count, self.abnormal_count)
        # debug check
        # if self.phase == "train":
        #     assert abnormal_count == 0, "%s\nAbnormal data not allowed in train dataset." % msg
        # if self.phase == "test":
        #     assert abnormal_count != 0, "%s\nNo abnormal data found in test test" % msg
        # pd.set_option("display.max_colwidth", None)
        # print("DF", self.phase, self.df[["image", "x1x2y1y2"]])

    @staticmethod
    def label_transform_default(image, label, anomaly_size_px):
        """ This label transform is designed for SAGAN.
        Return 0 for normal images and 1 for abnormal images """

        if label in ["NORMAL0", "NORMAL1"] or anomaly_size_px == 0:
            return ATZDataset.NORMAL
        else:
            return ATZDataset.ABNORMAL

    def __len__(self):
        return len(self.df)

    def get_meta(self, idx):
        record = self.df.iloc[idx, :]
        # read metadata
        current_file = record[["image"]].values[0]
        patch_id = record[["patch_id"]].values[0]
        label_txt = record[["label_txt"]].values[0]
        x1, x2, y1, y2 = ast.literal_eval(record[["x1x2y1y2"]].values[0])
        anomaly_size = record[["anomaly_size"]].values[0]
        label = record[["is_anamoly"]].values[0]
        # read image from cache
        img_patches = self.get_cached_image(current_file)
        img_p = img_patches[patch_id]
        # is_good = good_enough(img_p, label)
        # cv2.imshow("%s patch" % is_good, img_p)
        # pilimage = Image.fromarray(np.uint8(img_p))
        # pilimage.show("%s patch.PNG" % is_good)

        return dict(current_file=current_file, label_txt=label_txt, x1=x1, x2=x2, y1=y1, y2=y2,
                    anomaly_size=anomaly_size, is_anomaly=label, image_patch=img_p.copy(),
                    patch_id=patch_id,
                    #    mostly_dark=not is_good
                    )

    def __getitem__(self, idx):
        """Read a patch and return the item"""
        metadata = self.get_meta(idx)
        # read a record
        # record = self.df.iloc[idx, :]
        # read metadata
        # current_file = metadata["image"]
        # label_txt = metadata["label_txt"]
        # x1, x2, y1, y2 = metadata["x1"], metadata["x2"], metadata["y1"], metadata["y2"]
        # anomaly_size = metadata["anomaly_size"]
        label = metadata["is_anomaly"]
        # read image from cache
        image_patch = metadata['image_patch']
        pil = Image.fromarray(image_patch.astype('uint8'))
        # transform image
        if self.transform:
            tensor_img = self.transform(pil)
        else:
            tensor_img = transforms.ToTensor()(image_patch)
        # cv2.imshow("patch", image)
        return (tensor_img.to(self.device), torch.tensor(label, dtype=torch.uint8).to(self.device)), metadata

    def cache_limit_check(self):
        """
        Adjust and randomly clear cache
        """
        items = len(self.cache_image.keys())
        if items == ATZDataset.CACHE_ITEM_LIMIT:
            # cache full remove item
            self.cache_image.pop(random.choice(list(self.cache_image.keys())))

    def get_cached_image(self, current_file):
        """
            Read or update image cache. Returns an image object.
            return numpy image patches
        """
        # try to read image form cache
        img_patches = self.cache_image[current_file]
        if img_patches is None:
            # prepare image path
            img_path = os.path.join(self.img_dir, current_file)
            # read imagedata
            if self.nc == 3:
                image = cv2.imread(img_path)
            else:
                image = cv2.imread(img_path, 0)
            # convert to greyscale
            # image = image.convert("L")
            if self.gbl_wavelet_transform:
                image = self.gbl_wavelet_transform(image)

            # create patches
            emp = EMPatches()
            # print("patch_size", type(self.patch_size), self.patch_size)
            # print("patch_overlap", type(self.patch_overlap), self.patch_overlap)
            img_patches, indices = emp.extract_patches(image, patchsize=int(self.patch_size),
                                                       overlap=float(self.patch_overlap))
            # check cache size
            self.cache_limit_check()
            # save image to cache
            self.cache_image[current_file] = img_patches
        return img_patches
