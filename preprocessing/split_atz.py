import shutil
import random
from patchify import patchify
import pathlib
import os
import cv2
import matplotlib.pyplot as plt

from preprocessing import read_content


def copy_files(files, src_root, dest_root):
    for im_name in files:
        # objects are present
        im_src = str(src_root / im_name)
        im_dest = str(dest_root / im_name)
        shutil.copy(im_src, im_dest)


def create_dataset(src_dir, dataset_dir, is_normal):
    train_normal_path = dataset_dir / "train" / "0.normal"
    test_normal_path = dataset_dir / "test" / "0.normal"
    test_abnormal_path = dataset_dir / "test" / "1.abnormal"

    # cleanup
    shutil.rmtree(str(dataset_dir / "train"), ignore_errors=True)
    shutil.rmtree(str(dataset_dir / "test"), ignore_errors=True)
    os.makedirs(str(train_normal_path))
    os.makedirs(str(test_normal_path))
    os.makedirs(str(test_abnormal_path))

    # list images
    files = os.listdir(str(src_dir))

    # select image and annotation
    normal = []
    abnormal = []

    # separate files into normal and abnormal
    for image_name in files:
        if is_normal(image_name):
            normal.append(image_name)
        else:
            abnormal.append(image_name)

    # shuffle filenames
    random.seed(47)
    random.shuffle(normal)
    # split normal into train and test
    _80 = int(len(normal) * 0.8)
    train_split = normal[:_80]
    test_split = normal[_80:]
    # random.seed(47)
    # random.shuffle(abnormal)

    # copy abnormal files
    copy_files(abnormal, src_dir, test_abnormal_path)

    # copy normal train files
    copy_files(train_split, src_dir, train_normal_path)

    # copy normal test files
    copy_files(test_split, src_dir, test_normal_path)


# create a dataset for SAGAN from ATZ
img_root = pathlib.Path("/Users/soumen/Downloads/Datasets/Active Terahertz Imaging Dataset")
voc_root = img_root / "THZ_dataset_det_VOC/Annotations"


def check_normal(path):
    global voc_root
    voc = voc_root / path.replace(".jpg", ".xml")
    name, boxes = read_content(voc)
    return len(boxes) == 1


src = img_root / "THZ_dataset_det_VOC/JPEGImages"
atz_sagan_data_dir = pathlib.Path("/Users/soumen/Desktop/Skip-Attention-GAN/customdataset/atz")
create_dataset(src, atz_sagan_data_dir, check_normal)
