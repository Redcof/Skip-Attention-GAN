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

class_name_dict = dict(KK="Kitchen Knife", GA="Gun", MD="Metal Dagger",
                       SS="Scissors", WB="Water Bottle", CK="Ceramic Knife",
                       CP="Cell Phone", KC="Key Chain", LW="Leather Wallet",
                       CL="Cigarette Lighter", UN="Unknown", UNKNOWN="UNKNOWN",
                       HUMAN="HUMAN",
                       )
# do not change this order
atz_classes = ["NORMAL0", "NORMAL1", "HUMAN",
               "UNKNOWN", "UN",
               "CP", "MD", "GA", "KK", "SS",
               "KC", "WB", "LW", "CK", "CL"]
# index for a class from the `atz_classes` list. Any class upto 'SS'
# in that list will be ignored during dataset creation
atz_ignore_cls_idx_lim = atz_classes.index("WB")

root = pathlib.Path("/Users/soumen/Downloads/Datasets/Active Terahertz Imaging Dataset")
image_root = root / "THZ_dataset_det_VOC/JPEGImages"
voc_root = root / "THZ_dataset_det_VOC/Annotations"
display = False  # display plots
dataset_save_path = pathlib.Path("./patch_atz")
patch_size = 64
mask_iou = 0.3  # this will be used to decide anomaly or not
postfix = "_%d_%0.2f_%dx%d.csv" % (atz_ignore_cls_idx_lim + 1, mask_iou, patch_size, patch_size)
patch_dataset_csv = str(dataset_save_path / ("atz_patch_dataset_%s.csv" % postfix))

# create save directory
patch_image_save_path = dataset_save_path / "images"
os.makedirs(str(patch_image_save_path), exist_ok=True)

intersection = patch_size * patch_size


def create_patch_dataset():
    # select image and annotation
    image_files = os.listdir(image_root)
    atz_patch_dataset_df = pd.DataFrame()
    for image_name in tqdm(image_files):
        if not any([c in image_name for c in atz_classes[atz_ignore_cls_idx_lim + 1:]]):
            continue
        # image_name = "S_P_M5_MD_F_W_SS_V_W_back_0906085926.jpg"
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
            cls_idx = atz_classes.index(class_)
            if cls_idx <= atz_ignore_cls_idx_lim:
                # ignore classes
                continue
            mask[ymin:ymax + 1, xmin:xmax + 1] = cls_idx
            if display:
                x, y, w, h = xmin, ymin, abs(xmax - xmin), abs(ymax - ymin)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (r, g, b), 4)
                print("boxes:", class_, atz_classes.index(class_))

        if display:
            # render image
            cv2.imshow("image", img)
            cv2.imshow("mask", mask)
            print(np.unique(mask))

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
        # print(np.unique(mask))
        cols = 4
        rows = len(img_patches) // cols + 1
        for idx, (img_p, mask_p) in enumerate(zip(img_patches, mask_patches)):
            # ignore a patch with all [black pixels
            # img_p
            # create patch file name
            patch_file_name = "%s.jpg" % image_name.replace(".jpg", "_%04d" % idx)
            # class label
            max_val = np.max(mask_p)
            class_index = max_val
            # class name
            label_txt = atz_classes[int(class_index)]
            # if class_index > 0: print(label_txt, class_index, idx, np.unique(mask_p))
            # calculate IOU
            # the entire patch is the union,
            # and the masked area is intersection.
            mask_p[mask_p > 0] = 1  # replace all non zeros with 1
            union = np.sum(mask_p.flatten()) + 1e-20  # sum of 1s give mask area(the union)
            iou = intersection / union
            # compare iou
            if 1 >= iou >= mask_iou:
                # anomaly
                is_anomaly = True
            else:
                # normal
                is_anomaly = False
                label_txt = "NORMAL"

            # prepare record
            dictionary = dict(image=patch_file_name, patch_id=idx, label=class_index, label_txt=label_txt,
                              anomaly=is_anomaly)
            # update dataframe
            df_dictionary = pd.DataFrame([dictionary])
            atz_patch_dataset_df = pd.concat([atz_patch_dataset_df, df_dictionary], ignore_index=True)
            # save image
            cv2.imwrite(str(patch_image_save_path / patch_file_name), img_p)
            if display:
                ax = plt.subplot(rows, cols, idx + 1)
                ax.axis("off")
                mask_rgb = mask_p.copy().astype('uint8')
                mask_rgb[mask_rgb > 0] = 255
                ax.set_title("%d" % idx, fontdict=dict(color="red"))
                ax.imshow(np.hstack((img_p, mask_rgb)))

                # ax = plt.subplot(rows, cols, idx + 2)
                # ax.axis("off")
                # ax.imshow(mask_p)
        # save csv
        atz_patch_dataset_df.columns = ["image", "patch_id", "label", "label_txt", "anomaly"]
        atz_patch_dataset_df.to_csv(patch_dataset_csv)


if __name__ == '__main__':
    create_patch_dataset()
    print("Files are saved @", str(patch_image_save_path))
    print("Metadata @", patch_dataset_csv)
    if display:
        plt.show()
        cv2.destroyAllWindows()
