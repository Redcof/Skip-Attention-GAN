import numpy as np
import pandas as pd
from empatches import EMPatches
from patchify import patchify
import pathlib
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocessing.utils import read_content

atz_classes = ["UNKNOWN", "UN", "HUMAN",
               "KK", "GA", "MD",
               "SS", "WB", "CK",
               "CP", "KC", "LW", "CL"]
atz_ignore_classes = ["HUMAN", "MD"]

root = pathlib.Path("/Users/soumen/Downloads/Datasets/Active Terahertz Imaging Dataset")
image_root = root / "THZ_dataset_det_VOC/JPEGImages"
voc_root = root / "THZ_dataset_det_VOC/Annotations"
display = True
dataset_save_path = pathlib.Path("./patch_atz")
patch_size = 64
mask_iou = 0.3  # this will be used to
patch_dataset_csv = str(dataset_save_path / ("atz_patch_dataset_iou%d_%d.csv" % (mask_iou, patch_size)))
atz_patch_dataset_df = pd.DataFrame()

# create save directory
os.makedirs(str(dataset_save_path / "images"), exist_ok=True)


def create_patch_dataset():
    # select image and annotation
    image_files = os.listdir(image_root)
    for image_name in tqdm(image_files):
        image_name = "S_P_M5_MD_F_W_SS_V_W_back_0906085926.jpg"
        voc_xml = voc_root / image_name.replace(".jpg", ".xml")

        # read annotation
        name, boxes = read_content(str(voc_xml))
        # read image
        img = cv2.imread(str(image_root / image_name))
        # print(img.shape)
        # create a mask to store class info
        mask = np.zeros(img.shape)
        r, g, b = 255, 0, 0
        # apply bbox and create mask
        for box_info in boxes:
            (xmin, ymin, xmax, ymax, cx, cy, class_) = box_info
            if class_ in atz_ignore_classes:
                continue
            if display:
                x, y, w, h = xmin, ymin, abs(xmax - xmin), abs(ymax - ymin)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (r, g, b), 4)
                print("boxes:", class_, atz_classes.index(class_))
            mask[ymin:ymax + 1, xmin:xmax + 1] = atz_classes.index(class_)

        if display:
            # render image
            cv2.imshow("image", img)
            cv2.imshow("mask", mask)

        ## PATCHIFY
        # # create patches
        # patch_w, patch_h, step = 128, 128, (64, 64, 3)  # 67, 67, 30
        # print((patch_w, patch_h, 3), step)
        # patches = patchify(img, (patch_h, patch_w, 3), step=step)
        # print(patches.shape, img.shape)
        # rows = patches.shape[0]
        # cols = patches.shape[1]
        # plt.axis("off")
        # for r in range(0, rows):
        #     for c in range(0, cols):
        #         idx = (r * cols + c + 1)
        #         im = patches[r, c, 0, :, :]
        #         ax = plt.subplot(rows, cols, idx)
        #         ax.axis("off")
        #         plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        #         # cv2.imshow("%d"%idx, im)
        # plt.show()
        # cv2.destroyAllWindows()

        ## EMPatches
        # load module
        emp = EMPatches()
        img_patches, indices = emp.extract_patches(img, patchsize=patch_size, overlap=0.2)
        mask_patches, _ = emp.extract_patches(mask, patchsize=patch_size, overlap=0.2)

        cols = 4
        rows = len(img_patches) // cols + 1
        for idx, (img_p, mask_p) in enumerate(zip(img_patches, mask_patches)):
            # create patch file name
            patch_file_name = "%s.jpg" % image_name.replace(".jpg", "%04d" % idx)
            # class name
            class_label = np.max(mask_p)
            # calculate IOU
            mask_p[mask_p > 0] = 1  # replace all non zeros with 1
            a1 = np.sum(mask_p.flatten())
            a2 = patch_size * patch_size
            iou = a1 / a2
            if iou >= mask_iou:
                # anomaly
                class_label
                ...
            else:
                # normal
                ...
            if display:
                ax = plt.subplot(rows, cols, idx + 1)
                ax.axis("off")
                mask_rgb = mask_p.copy().astype('uint8')
                mask_rgb[mask_rgb > 0] = 255
                ax.imshow(np.hstack((img_p, mask_rgb)))

                # ax = plt.subplot(rows, cols, idx + 2)
                # ax.axis("off")
                # ax.imshow(mask_p)
        return


if __name__ == '__main__':
    create_patch_dataset()
    if display:
        plt.show()
        cv2.destroyAllWindows()
