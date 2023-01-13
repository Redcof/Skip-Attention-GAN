import os.path

import cv2
import numpy as np
import pywt
from matplotlib import pyplot as plt
from skimage.restoration import estimate_sigma, denoise_wavelet


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


# https://www.kaggle.com/code/theoviel/denoising-with-direct-wavelet-transform
def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def universal_threshold(x, coeff, level):
    # D.L. Donoho and I.M. Johnstone. Ideal Spatial Adaptation via Wavelet Shrinkage. Biometrika.
    # Vol. 81, No. 3, pp.425-455, 1994.
    # DOI:10.1093/biomet/81.3.425
    # https://www.hindawi.com/journals/mpe/2015/280251/
    # universal threshold lambda = sigma*sqrt(2 log(N))
    # where
    # sigma: is the average variance of the noise and
    # N: is the signal length.
    sigma = madev(coeff[-level]) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    return uthresh


def wavelet_denoise_rgb(image, wavelet='bior4.4', method='VisuShrink', channel_axis=2, decomposition_level=2,
                        threshold_mode='soft'):
    # D.L. Donoho and I.M. Johnstone. Ideal Spatial Adaptation via Wavelet Shrinkage. Biometrika.
    # Vol. 81, No. 3, pp.425-455, 1994.
    # DOI:10.1093/biomet/81.3.425
    # https://www.hindawi.com/journals/mpe/2015/280251/
    sigma = estimate_sigma(image, average_sigmas=True, channel_axis=channel_axis)
    denoised_x = denoise_wavelet(image, sigma=sigma, wavelet=wavelet, mode=threshold_mode,
                                 wavelet_levels=decomposition_level, channel_axis=channel_axis,
                                 convert2ycbcr=True, method=method,
                                 rescale_sigma=True)
    return denoised_x


if __name__ == '__main__':
    root = "/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset/THZ_dataset_det_VOC/JPEGImages"
    file = "T_P_M6_MD_F_LL_CK_F_C_WB_F_RT_front_0906154134.jpg"
    img_bgr = cv2.imread(os.path.join(root, file))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    wavelets = ['sym20', 'bior4.4', 'db']
    plt.show()
    print(pywt.wavelist())
    cols = 4
    rows = (len(wavelets) + 1) // cols
    ax = plt.subplot(rows, cols, 1)
    plt.imshow(img_rgb)
    ax.set_title("Original")
    for idx, wv in enumerate(wavelets):
        try:
            denos_img = wavelet_denoise_rgb(img_rgb.copy(), wavelet=wv, decomposition_level=20)
            ax = plt.subplot(rows, cols, idx + 2)
            plt.imshow(denos_img)
            ax.set_title(wv)
        except:
            ...
    plt.show()
