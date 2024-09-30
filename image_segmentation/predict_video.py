import time

import cv2
import numpy as np
import tensorflow as tf

from constants import MODEL_NAME, MODEL_DIR, CLASSES, COLORS, IMAGES_SIZE1, IMAGES_SIZE2, resized_images

FRAME_RATE = 15
FRAME_LENGTH = 30
VIDEO_TEST_PATH = "./videos/videoFile.mp4"
VIDEO_TEST_SAVE = f"./videos/videoFile{'_rs' if resized_images else ''}_{MODEL_NAME}2.mp4"
MODEL_TEST_NAME = MODEL_DIR + MODEL_NAME + ".h5"


# MODEL_TEST_NAME = MODEL_DIR + "mobile_small" + ".h5"

def pass_through_video():
    start_frame = 0
    framerate = FRAME_RATE

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    frame_index = 0

    out = cv2.VideoWriter(
        VIDEO_TEST_SAVE,
        cv2.VideoWriter_fourcc(*'mp4v'),
        framerate,
        # (frame_width, frame_height))
        # (int(width), int(height))
        (IMAGES_SIZE1, IMAGES_SIZE2)
    )

    while cap.isOpened(): # and frame_index < FRAME_LENGTH:
        start = time.time()
        ret, frame = cap.read()

        if ret == True:
            frame = cv2.resize(frame, (IMAGES_SIZE1, IMAGES_SIZE2))  # Resize to match model input shape

            pred_mask = model.predict(np.expand_dims(np.array(frame), axis=0))
            mask = np.argmax(pred_mask, axis=-1)[0]
            confidences = np.amax(pred_mask, axis=-1)[0]

            for id in range(CLASSES - 1):
                coords = np.where(mask == id + 1)
                x_coords = coords[0]
                y_coords = coords[1]
                selected_confidences = confidences[mask == id + 1]

                if len(selected_confidences) > 0:
                    confidence = np.amax(selected_confidences)
                    confidence_id = np.argmax(selected_confidences)
                    thickness = -1
                    radius = 5
                    center_coordinates = (y_coords[confidence_id], x_coords[confidence_id])
                    frame = cv2.circle(frame, center_coordinates, radius, COLORS[id].tolist(), thickness)

                    fontScale = 0.2
                    text = str(confidence * 100)[:2]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    color = (0, 0, 0)
                    thickness = 1
                    txt_size = cv2.getTextSize(text, font, fontScale, thickness)
                    end_x = center_coordinates[0] - txt_size[0][0] // 2
                    end_y = center_coordinates[1] + txt_size[0][1] // 2

                    frame = cv2.putText(frame, text, (end_x, end_y), font, fontScale, color, thickness, cv2.LINE_AA)

            out.write(frame)
        else:
            break

        frame_index += 1
        print(f"Frame {frame_index}: time {time.time() - start:.2f}s")

    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()


# -----------------------MAIN--------------------
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

cap = cv2.VideoCapture(VIDEO_TEST_PATH)

model = tf.keras.models.load_model(MODEL_TEST_NAME)

# print(model.summary())


# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

pass_through_video()