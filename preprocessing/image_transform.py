import os.path
import pathlib
import platform

import cv2
import numpy as np
import pywt
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
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
    # CONCEPT: https://www.hindawi.com/journals/mpe/2015/280251/
    # universal threshold lambda = sigma*sqrt(2 log(N))
    # where
    # sigma: is the average variance of the noise and
    # N: is the signal length.
    sigma = madev(coeff[-level]) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    return uthresh


def wavelet_denoise_rgb(image, channel_axis, wavelet='bior4.4', method='VisuShrink',
                        decomposition_level=2, threshold_mode='soft', psnr=False):
    # CODE: https://www.exptech.co.in/2021/03/in-this-video-wavelet-transform-based.html
    sigma = estimate_sigma(image, average_sigmas=True, channel_axis=channel_axis)
    deno_scaled_img = denoise_wavelet(image, sigma=sigma, wavelet=wavelet, mode=threshold_mode,
                                      wavelet_levels=decomposition_level, channel_axis=channel_axis,
                                      convert2ycbcr=True, method=method,
                                      rescale_sigma=True)
    # deno_scaled_img values between [0-1]
    # rescale back the de-noised image to [0-255] space
    # diff = np.max(deno_scaled_img) - np.min(deno_scaled_img)
    # diff = 0.0000000001 if int(diff) == 0 else diff
    # deno_image = ((deno_scaled_img - np.min(deno_scaled_img)) * 255) / diff
    deno_image = cv2.normalize(deno_scaled_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    if psnr:
        psnr = peak_signal_noise_ratio(image, deno_image)
        return deno_image, psnr
    else:
        return deno_image


def enhance_contrast_by_wavelets(image, contrast_factor, wavelet='db1', decomp_level=1):
    # Perform the wavelet transform
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=decomp_level)

    # Adjust the amplitude of the high frequency coefficients
    coeffs[-1] = coeffs[-1] * contrast_factor

    # Perform the inverse wavelet transform
    img_back = pywt.waverec2(coeffs, wavelet=wavelet)

    return img_back


def enhance_contrast_by_fourier(image, contrast_factor):
    # code generated by chat-gpt
    # https://chat.openai.com/chat/e6267f99-bad1-4ff8-9082-969c10168288

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform the Fourier transform
    f = np.fft.fft2(gray)

    # Shift the zero-frequency component to the center of the spectrum
    fshift = np.fft.fftshift(f)

    # Get the magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Adjust the amplitude of the frequency components
    fshift = fshift * (1 + contrast_factor)

    # Shift the zero-frequency component back to the original position
    f_ishift = np.fft.ifftshift(fshift)

    # Perform the inverse Fourier transform
    img_back = np.fft.ifft2(f_ishift)

    # Get the real part of the result
    img_back = np.real(img_back)

    # Normalize the result
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return img_back


if __name__ == '__main__':
    if platform.system() == "Darwin":
        root = pathlib.Path(
            "/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset/THZ_dataset_det_VOC/JPEGImages")
    elif platform.system() == "Windows":
        root = pathlib.Path(r"C:\Users\dndlssardar\Downloads\THZ_dataset_det_VOC\JPEGImages")

    # files
    files = ["S_N_M2_SS_F_C_MD_V_W_back_0904091058.jpg", "T_P_M6_MD_F_LL_CK_F_C_WB_F_RT_back_0906154134.jpg"]

    # wavelet settings
    d = {'wavelet': 'sym4', 'method': 'VisuShrink', 'level': 1, 'mode': 'soft'}
    decomp_ = [1, 3]  # 3
    method_ = ["VisuShrink"]  # "BayesShrink",
    mode_ = ["soft", "hard"]  # "hard"
    wavelets = ['bior6.8', 'coif17']  # 'bior4.4'

    print(pywt.wavelist())
    cols = 6
    rows = (len(wavelets) * len(mode_) * len(method_) * len(decomp_) * len(files) + len(files))
    rows = rows // cols if rows % cols == 0 else rows // cols + 1
    rows = 1 if rows < 1 else rows
    plt.figure(figsize=[cols * 2, rows * 2])

    idx = 1
    for file in files:
        img_bgr = cv2.imread(str(root / file))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        ax = plt.subplot(rows, cols, idx)
        idx += 1
        plt.imshow(img_rgb)
        ax.set_title("Original")
        for lv in decomp_:
            for wv in wavelets:
                for mo in mode_:
                    for me in method_:
                        try:
                            # denos_img, psnr = wavelet_denoise_rgb(img_rgb.copy(),
                            #                                       channel_axis=2,
                            #                                       wavelet=wv,
                            #                                       method=me,
                            #                                       threshold_mode=mo,
                            #                                       decomposition_level=lv, psnr=True)
                            denos_img = enhance_contrast_by_wavelets(img_rgb.copy(), 0.2)
                            psnr = peak_signal_noise_ratio(img_rgb, denos_img)
                            ax = plt.subplot(rows, cols, idx)
                            idx += 1
                            txt = "%s,%s,\n%s,%d, %f" % (wv, mo, me, lv, psnr)
                            # cv2.imshow(txt, denos_img)
                            plt.imshow(denos_img)
                            ax.set_title(txt, fontdict=dict(fontsize=10))

                        except:
                            ...
    plt.show()
