import numpy as np
import pywt
from matplotlib import pyplot as plt


def w2d(img_8C1, mode='haar', level=1, display=False):
    # Datatype conversions
    # convert to grayscale
    # imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    # convert to float
    img_8C1 = np.float32(img_8C1)
    img_8C1 /= 255
    # compute coefficients
    coeffs = pywt.wavedec2(img_8C1, mode, level=level)

    # Process Coefficients
    coeffs_h = list(coeffs)
    coeffs_h[0] *= 0

    # reconstruction
    reconst_img = pywt.waverec2(coeffs_h, mode)
    reconst_img *= 255
    reconst_img = np.uint8(reconst_img)
    if display:
        # Display result
        plt.figure(figsize=[10, 10])
        plt.subplot(121)
        plt.imshow(img_8C1, cmap="gray")
        ax = plt.subplot(122)
        ax.set_title(mode)
        plt.imshow(reconst_img, cmap="gray")
        # cv2.imshow('image',reconst_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        plt.show()
    return reconst_img


def mat2grey(x, display=False):
    # https://en.wikipedia.org/wiki/Feature_scaling
    # Rescaling (min-max normalization)
    a, b = 0, 255
    minx, maxx = np.min(x), np.max(x)
    denormalised = a + (((x - minx) * (b - a)) / (maxx - minx)).astype('uint8')
    if display:
        plt.imshow(denormalised, cmap="gray")
        plt.show()
    return denormalised
