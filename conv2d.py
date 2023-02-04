# 2D Convolution
import numpy as np

def Conv2D(image, kernel, padding=0, stride=1):

    # cross correlation
    kernel = np.flipud(np.fliplr(kernel))

    (m, n) = image.shape
    (p,q) = kernel.shape

    output = np.zeros([(m + 2*padding - p)//stride + 1, (n + 2*padding - q)//stride + 1])

    padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    (m, n) = padded_image.shape

    for i in range(0, m + 2*padding - p + 1, stride):
        for j in range(0, n + 2*padding - q + 1, stride):
            output[i//stride, j//stride] = np.sum(kernel * padded_image[i:i+p, j:j+q])

    return output
