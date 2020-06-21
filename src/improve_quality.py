import numpy as np
import cv2
import sys


def multiscale_retinex(image):
    """
        Using the Multiscale Retinex with Color Restoration for image enhancement
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.52.1669&rep=rep1&type=pdf
        http://ieeexplore.ieee.org/document/597272/
        http://ieeexplore.ieee.org/document/6176791/
        http://www.ipol.im/pub/art/2014/107/

    :param image:
        Original Image with shape (H, W, C)
    :return:
    """
    if len(image.shape) != 3:
        sys.exit('Image need have shape (H, W, C)')
    H, W, C = image.shape
    print('\t> Improving quality ...')
    image = cv2.resize(image, (int(H / 2), int(W / 2)), interpolation=cv2.INTER_AREA)
    sigma = [15, 80, 250]
    low_clip = 0.01
    high_clip = 0.99
    image = image.astype(np.float64) + 1.0

    intensity = np.sum(image, axis=2) / image.shape[2]

    # Multi-scale Retinex
    retinex = np.zeros_like(intensity)
    for s in sigma:
        retinex += np.log10(intensity) - np.log10(cv2.GaussianBlur(intensity, (0, 0), s))
    retinex /= len(sigma)

    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)

    # Simplest color balance
    total = retinex.shape[0] * retinex.shape[1]
    for c in range(retinex.shape[2]):
        unique, counts = np.unique(retinex[:, :, c], return_counts=True)
        current, high_val, low_val = 0, 0, 0
        for u, count in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += count
        retinex[:, :, c] = np.maximum(np.minimum(retinex[:, :, c], high_val), low_val)

    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex)) * 255.0 + 1.0
    out = np.zeros_like(image)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            B = np.max(image[y, x])
            A = np.minimum(256.0 / B, retinex[y, x, 0] / intensity[y, x, 0])
            # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#filter2d
            out[y, x, 0] = A * image[y, x, 0]
            out[y, x, 1] = A * image[y, x, 1]
            out[y, x, 2] = A * image[y, x, 2]

    out = cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)
    print('\t> End improving quality')
    return np.uint8(out - 1.0)
