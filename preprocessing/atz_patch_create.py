import numpy as np
import pandas as pd
from empatches import EMPatches
from patchify import patchify
import pathlib
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocessing.utils import read_vocxml_content, parsing_filename

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
# in that list will be ignored during dataset creation, set -1 to consider all classes
atz_ignore_cls_idx_lim = atz_classes.index("HUMAN")

root = pathlib.Path("/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset")
image_root = root / "THZ_dataset_det_VOC/JPEGImages"
voc_root = root / "THZ_dataset_det_VOC/Annotations"
display = False  # display plots
dataset_save_path = pathlib.Path("../customdataset/atz")
patch_size = 128  # 64, 128

# create save directory
patch_image_save_path = dataset_save_path / "images"
os.makedirs(str(patch_image_save_path), exist_ok=True)

# inter
intersection = patch_size * patch_size


def create_patch_dataset():
    # select image and annotation
    image_files = os.listdir(image_root)
    atz_patch_dataset_df = pd.DataFrame()
    for image_name in tqdm(image_files):
        if not any([c in image_name for c in atz_classes[atz_ignore_cls_idx_lim + 1:]]):
            continue
        # image_name = 'S_P_F1_WB_F_LL_CK_V_LL_back_0907094821.jpg'
        voc_xml = voc_root / image_name.replace(".jpg", ".xml")
        base, data = parsing_filename(str(voc_root), image_name.replace(".jpg", ".xml"))

        subject_gender = base['subject_gender']
        subject_id = base['subject_id']
        threat_present = base['presence']
        front_back = base['front_back']

        # read annotation
        name, boxes = read_vocxml_content(str(voc_xml))
        # read image
        img = cv2.imread(str(image_root / image_name))
        # print(img.shape)
        # create a mask to store class info
        mask = np.zeros(img.shape)
        r, g, b = 255, 0, 0
        box_dict = dict()
        # apply bbox and create mask
        for box_info in boxes:
            xmin, ymin, xmax, ymax, cx, cy, label_txt = box_info
            cls_idx = atz_classes.index(label_txt)
            box_dict[label_txt] = xmin, ymin, xmax, ymax
            if cls_idx <= atz_ignore_cls_idx_lim or label_txt == "HUMAN":
                # ignore classes
                continue
            mask[ymin:ymax + 1, xmin:xmax + 1] = cls_idx
            if display:
                x, y, w, h = xmin, ymin, abs(xmax - xmin), abs(ymax - ymin)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (r, g, b), 4)
                print("boxes:", label_txt, atz_classes.index(label_txt))

        if display:
            # render image
            cv2.imshow("image", img)
            cv2.imshow("mask", mask)
            print(np.unique(mask))

        emp = EMPatches()
        img_patches, indices = emp.extract_patches(img, patchsize=patch_size, overlap=0.2)
        mask_patches, _ = emp.extract_patches(mask, patchsize=patch_size, overlap=0.2)
        # print(np.unique(mask))
        cols = 4
        rows = len(img_patches) // cols + 1
        dictionary_ls = []
        for idx, (img_p, mask_p, patch_loc) in enumerate(zip(img_patches, mask_patches, indices)):
            img_p = img_p.copy()
            mask_p = mask_p.copy()
            # ignore a patch with all [black pixels]
            # img_p
            # create patch file name
            # patch_file_name = "%s.jpg" % image_name.replace(".jpg", "_%04d" % idx)
            # class label
            max_val = int(np.max(mask_p))
            class_index = max_val

            # class name
            label_txt = atz_classes[class_index]
            global_box = box_dict.get(label_txt, (0, 0, 0, 0))
            mask_p[mask_p > 0] = 1  # replace all non zeros with 1
            # object area in musk
            obj_area_px = np.sum(mask_p.flatten())

            dictionary = dict(image=image_name,
                              threat_present=threat_present,
                              front_back=front_back,
                              patch_id=idx,
                              label=class_index,
                              label_txt=label_txt,
                              global_x1y1x2y2=global_box,
                              anomaly_size=obj_area_px, x1x2y1y2=patch_loc,
                              subject_gender=subject_gender,
                              subject_id="%s%s" % (subject_gender, subject_id))
            # prepare record
            dictionary_ls.append(dictionary)
            # save image
            # cv2.imwrite(str(patch_image_save_path / patch_file_name), img_p)
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
        # update dataframe
        df_dictionary = pd.DataFrame(dictionary_ls)
        atz_patch_dataset_df = pd.concat([atz_patch_dataset_df, df_dictionary], ignore_index=True)
    # save csv
    postfix = "_%d_%d_%d" % (atz_ignore_cls_idx_lim + 1, patch_size, len(indices))
    patch_dataset_csv = str(dataset_save_path / ("atz_patch_dataset_%s.csv" % postfix))
    atz_patch_dataset_df.columns = dictionary.keys()
    atz_patch_dataset_df.to_csv(patch_dataset_csv)
    # print("Files are saved @", str(patch_image_save_path))
    print("Metadata @", patch_dataset_csv)


if __name__ == '__main__':
    for i in [128]:
        patch_size = i
        create_patch_dataset()
    if display:
        plt.show()
        cv2.destroyAllWindows()
