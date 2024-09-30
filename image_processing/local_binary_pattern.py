# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to calculate LBP for a given image
def local_binary_pattern(image, radius=1):
    rows, cols = image.shape

    lbp_image = np.zeros_like(image, dtype=np.uint8)

    # neighborhood offset (3x3 grid)
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center_pixel = image[i, j]
            binary_string = ''

            # Compare center pixel with its neighbors
            for n in neighbors:
                neighbor_pixel = image[i + n[0], j + n[1]]
                binary_string += '1' if neighbor_pixel >= center_pixel else '0'

            # Convert the binary string to a decimal value (LBP value)
            lbp_value = int(binary_string, 2)

            # Assign the LBP value to the output image
            lbp_image[i, j] = lbp_value

    return lbp_image


if __name__ == '__main__':
    image = cv2.imread('./example.png', cv2.IMREAD_GRAYSCALE)

    lbp_image = local_binary_pattern(image)

    # Plot the image and LBP
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(lbp_image, cmap='gray')
    plt.title('LBP Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Histogram of LBP values
    lbp_hist, lbp_bins = np.histogram(lbp_image.ravel(), bins=np.arange(0, 256), range=(0, 256))

    plt.figure(figsize=(8, 4))
    plt.bar(lbp_bins[:-1], lbp_hist, width=0.5, color='gray', edgecolor='black')
    plt.title('Histogram of LBP')
    plt.xlabel('LBP value')
    plt.ylabel('Frequency')
    plt.show()