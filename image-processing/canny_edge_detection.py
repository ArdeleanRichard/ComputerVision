import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.signal import correlate2d




def scale(x):
    return (x - x.min()) / (x.max() - x.min()) * 255


def nms(G, theta):
    """Non Max Suppression"""

    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.int32)  # resultant image
    angle = theta * 180.0 / np.pi  # max -> 180, min -> -180
    angle[angle < 0] += 180  # max -> 180, min -> 0

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (22.5 + 135 <= angle[i, j] <= 22.5 + 180):
                r = G[i, j - 1]
                q = G[i, j + 1]

            elif 22.5 <= angle[i, j] < 22.5 + 45:
                r = G[i - 1, j + 1]
                q = G[i + 1, j - 1]

            elif 22.5 + 45 <= angle[i, j] < 22.5 + 45 + 90:
                r = G[i - 1, j]
                q = G[i + 1, j]

            elif 22.5 + 45 <= angle[i, j] < 22.5 + 135:
                r = G[i + 1, j + 1]
                q = G[i - 1, j - 1]

            if (G[i, j] >= q) and (G[i, j] >= r):
                Z[i, j] = G[i, j]
            else:
                Z[i, j] = 0
    return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):

    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = 25
    strong = 255

    strong_i, strong_j = np.where(img >= highThreshold)
    # zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res

def hysteresis(img, weak=0, strong=255):
    M, N = img.shape

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i, j] == weak):
                if (
                    (img[i+1, j-1] == strong) or (img[i+1, j] == strong) or
                    (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or
                    (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or
                    (img[i-1, j] == strong) or (img[i-1, j+1] == strong)
                ):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img


def canny_edge_detection(img):
    # 1. Noise reduction using Gaussian filter
    img_blur = cv.GaussianBlur(src=img, ksize=(5, 5), sigmaX=3)

    # 2. Gradient calculation with Sobel
    gradientX = correlate2d(img_blur, Kx)
    gradientY = correlate2d(img_blur, Ky)

    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(gradientY, cmap='gray')
    # plt.title('GradientY')
    # plt.subplot(122)
    # plt.imshow(gradientX, cmap='gray')
    # plt.title('GradientX')

    # fig.set_figwidth(8)

    G = scale(np.hypot(gradientX, gradientY))

    # plt.title('Sobel Edge Detection Result')
    # plt.imshow(G, cmap='gray')

    EPS = np.finfo(float).eps  # used to tackle the division by zero error
    theta = np.arctan(gradientY / (gradientX + EPS))
    # theta = np.arctan2(Gradient_Y, Gradient_X)

    # 3. Non max suppression
    img_nms = nms(G, theta)

    # 4. Double threshold
    img_dt = threshold(img_nms, lowThresholdRatio=0.05, highThresholdRatio=0.1)

    # 5. Hysteresis
    img_hyst = hysteresis(img_dt)

    return img_hyst


if __name__ == '__main__':
    Kx = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]], np.float32
    )

    Ky = np.array(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]], np.float32
    )

    img = cv.imread('./example.png', cv.IMREAD_GRAYSCALE)

    edges = cv.Canny(img, 100, 200)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(edges, cmap='gray')
    plt.title('OpenCV Canny Edge Detection'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(canny_edge_detection(img), cmap='gray')
    plt.title('OpenCV Canny Edge Detection'), plt.xticks([]), plt.yticks([])
    plt.show()

