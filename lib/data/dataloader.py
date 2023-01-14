"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import ast
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder

from customdataset.atzdataset import ATZDataset
from lib.data.datasets import get_cifar_anomaly_dataset
from lib.data.datasets import get_mnist_anomaly_dataset
from preprocessing.wavelet_transform import wavelet_denoise_rgb


class Data:
    """ Dataloader containing train and valid sets.
    """

    def __init__(self, train, valid):
        self.train = train
        self.valid = valid


##
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)

    ## CIFAR
    if opt.dataset in ['cifar10']:
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_ds = CIFAR10(root='./data', train=True, download=True, transform=transform)
        valid_ds = CIFAR10(root='./data', train=False, download=True, transform=transform)
        train_ds, valid_ds = get_cifar_anomaly_dataset(train_ds, valid_ds, train_ds.class_to_idx[opt.abnormal_class])

    ## MNIST
    elif opt.dataset in ['mnist']:
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        train_ds = MNIST(root='./data', train=True, download=True, transform=transform)
        valid_ds = MNIST(root='./data', train=False, download=True, transform=transform)
        train_ds, valid_ds = get_mnist_anomaly_dataset(train_ds, valid_ds, int(opt.abnormal_class))
    ## ATZ
    elif opt.dataset in ['atz']:
        assert os.path.exists(opt.atz_patch_db)  # mandatory for ATZ

        patch_dataset_csv = opt.atz_patch_db
        train_dataset_txt = opt.atz_train_txt
        test_dataset_txt = opt.atz_test_txt
        atz_ablation = opt.atz_ablation
        device = torch.device("cuda:0" if opt.device != 'cpu' else "cpu")

        NORMAL_CLASSES = ["NORMAL0", "NORMAL1"]
        try:
            atz_classes = ast.literal_eval(opt.atz_classes)
        except ValueError:
            atz_classes = []
        atz_classes.extend(NORMAL_CLASSES)

        try:
            atz_subjects = ast.literal_eval(opt.atz_subjects)
        except ValueError:
            atz_subjects = []

        try:
            atz_wavelet = ast.literal_eval(opt.atz_wavelet)
        except ValueError:
            atz_wavelet = {'wavelet': 'sym4', 'method': 'VisuShrink', 'level': 1, 'mode': 'soft'}

        object_area_threshold = opt.area_threshold  # 10%
        patchsize = opt.isize
        PATCH_AREA = patchsize ** 2

        def wavelet_transform(x):
            is_converted = False
            if isinstance(x, Image.Image):
                # if PIL convert to numpy
                is_converted = True
                x = np.array(x)

            x = wavelet_denoise_rgb(x,
                                    channel_axis=2,
                                    wavelet=atz_wavelet['wavelet'],  # 'sym4'
                                    method=atz_wavelet['method'],  # 'VisuShrink',
                                    decomposition_level=atz_wavelet['level'],  # 1
                                    threshold_mode=atz_wavelet['mode']  # 'hard'
                                    )

            if is_converted:
                # restore back to PIL if converted previously
                x = Image.fromarray(x)
            return x

        def label_transform(image, label, anomaly_size_px):
            """ This label transform is designed for SAGAN.
            Return 0 for normal images and 1 for abnormal images """
            normal = 0
            abnormal = 1
            # object area in patch must be bigger than some threshold
            if anomaly_size_px > 0:
                object_area_percent = anomaly_size_px / PATCH_AREA
            else:
                object_area_percent = 0

            if (label in NORMAL_CLASSES
                    or anomaly_size_px == 0
                    # not in iou range
                    or not (1 >= object_area_percent >= object_area_threshold)):
                return normal
            else:
                return abnormal

        transform = transforms.Compose([
            transforms.Resize((opt.isize, opt.isize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        train_ds = ATZDataset(patch_dataset_csv, opt.dataroot, "train",
                              atz_dataset_train_or_test_txt=train_dataset_txt,
                              device=device,
                              classes=atz_classes,
                              patch_size=opt.isize,
                              patch_overlap=opt.atz_patch_overlap,
                              subjects=atz_subjects,
                              ablation=atz_ablation,
                              transform=transform,
                              random_state=opt.manualseed,
                              label_transform=label_transform,
                              global_wavelet_transform=wavelet_transform if opt.atz_wavelet_denoise else None)
        valid_ds = ATZDataset(patch_dataset_csv, opt.dataroot, "test",
                              atz_dataset_train_or_test_txt=test_dataset_txt,
                              device=device,
                              classes=atz_classes,
                              patch_size=opt.isize,
                              patch_overlap=opt.atz_patch_overlap,
                              subjects=atz_subjects,
                              ablation=atz_ablation,
                              transform=transform,
                              random_state=opt.manualseed,
                              label_transform=label_transform,
                              global_wavelet_transform=wavelet_transform if opt.atz_wavelet_denoise else None)
        opt.log("Dataset '%s' => Normal:Abnormal = %d:%d" % ("train", train_ds.normal_count, train_ds.abnormal_count))
        opt.log("Dataset '%s' => Normal:Abnormal = %d:%d" % ("test", valid_ds.normal_count, valid_ds.abnormal_count))

    # FOLDER
    else:
        transform = transforms.Compose([transforms.Resize((opt.isize, opt.isize)),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

        train_ds = ImageFolder(os.path.join(opt.dataroot, 'train'), transform)
        valid_ds = ImageFolder(os.path.join(opt.dataroot, 'test'), transform)

    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=False, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)
