# Histogram Equalization to increase contrast
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def plot_hist(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()

path = "./example.png"
img = cv.imread(path, cv.IMREAD_GRAYSCALE)
plot_hist(img)

img_eq = cv.equalizeHist(img)
plot_hist(img_eq)
