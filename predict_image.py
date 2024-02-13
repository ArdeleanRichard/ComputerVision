import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras

from util import display_images
from constants import MODEL_DIR, IMAGES_SIZE1, IMAGES_SIZE2, IMAGES_DIR, MASKS_DIR, MODEL_NAME


def detect():
    # Load and preprocess the image
    image_name = "img2.png"
    image_path = IMAGES_DIR + image_name
    mask_path = MASKS_DIR + f"mask_{image_name}"
    image = cv2.imread(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Load the trained model
    model = keras.models.load_model(MODEL_DIR + "mobile_small.h5")

    # Generate predictions
    start = time.time()
    predictions = model.predict(image)
    print(f"Time to predict: {time.time() - start}s")
    predictions = np.argmax(predictions, axis=-1).astype(np.uint8)  # Threshold the predictions

    # Visualize the original image and the predicted mask
    display_images([image[0], mask[:, :, None], predictions[0][:, :, None]])



def test_pred_speed():
    # Load and preprocess the image
    # Load the trained model
    model = keras.models.load_model(MODEL_DIR + MODEL_NAME + ".h5")

    batch = 32
    images = np.zeros((32, IMAGES_SIZE1, IMAGES_SIZE2, 3))
    for id, image_path in enumerate(os.listdir(IMAGES_DIR)):
        image = cv2.imread(IMAGES_DIR + image_path)
        images[id] = image
        if id > 30:
            break

    # Generate predictions
    start = time.time()
    predictions = model.predict(images)
    print(f"Time to predict batch of {batch}: {time.time() - start}s")
    predictions = np.argmax(predictions, axis=-1).astype(np.uint8)  # Threshold the predictions


detect()
# test_pred_speed()