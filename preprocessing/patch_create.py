from patchify import patchify
import pathlib
import os
import cv2
import matplotlib.pyplot as plt

from preprocessing import read_content


def bounding_box_present(idx, patch_w, patch_h, ):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(img_hsv, (0, 50, 20), (5, 255, 255))
    mask2 = cv2.inRange(img_hsv, (175, 50, 20), (180, 255, 255))
    # Binary mask with pixels matching the color threshold in white
    mask = cv2.bitwise_or(mask1, mask2)

    # Determine if the color exists on the image
    if cv2.countNonZero(mask) > 0:
        print('Red is present!')
        return True
    else:
        print('Red is not present!')
        return False


root = pathlib.Path("/Users/soumen/Downloads/Datasets/Active Terahertz Imaging Dataset")
image_root = root / "THZ_dataset_det_VOC/JPEGImages"
voc_root = root / "THZ_dataset_det_VOC/Annotations"

# select image and annotation
image_name = "T_P_M6_LW_V_LL_CL_V_LA_SS_V_B_back_0906154716.jpg"
voc = voc_root / image_name.replace(".jpg", ".xml")

# read annotation
name, boxes = read_content(str(voc))
# read image
img = cv2.imread(str(image_root / image_name))
# apply bbox
r, g, b = 255, 0, 0
for box_info in boxes:
    (xmin, ymin, xmax, ymax, cx, cy, class_) = box_info
    x, y, w, h = xmin, ymin, abs(xmax - xmin), abs(ymax - ymin)
    if class_ == "HUMAN":
        continue
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (b, g, r), 4)
    print("boxes:", class_)

# render image
cv2.imshow(image_name, img)
# create patches
patch_w, patch_h, step = 67, 80, (34, 40, 3)  # 67, 67, 30
print((patch_w, patch_h, 3), step)
patches = patchify(img, (patch_h, patch_w, 3), step=step)
print(patches.shape, img.shape)

rows = patches.shape[0]
cols = patches.shape[1]
plt.axis("off")
for r in range(0, rows):
    for c in range(0, cols):
        idx = (r * cols + c + 1)
        im = patches[r, c, 0, :, :]
        ax = plt.subplot(rows, cols, idx)
        ax.axis("off")
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        # cv2.imshow("%d"%idx, im)
plt.show()
cv2.destroyAllWindows()
