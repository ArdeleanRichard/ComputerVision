import numpy as np

# Set the input and mask directories
resized_images = True
START_DIR = './dataset/start/'
IMAGES_DIR = './dataset/images/' if resized_images==False else './dataset/rs_images/'
MASKS_DIR = './dataset/masks/' if resized_images==False else './dataset/rs_masks/'
MODEL_DIR = './models/' if resized_images==False else './rs_models/'
MODEL_NAME = 'mobile_large' # eff_b2

IMAGES_SIZE1 = 480 if resized_images==False else 224
IMAGES_SIZE2 = 640 if resized_images==False else 224
OG_SIZE1 = 480
OG_SIZE2 = 640

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.01
DROPOUT = 0.3

TRAIN_DATA_SIZE = 9600


CLASSES = 17
if resized_images == False:
    weight = 1000.0
    CLASS_WEIGHTS = [1.0,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     ]
else:
    weight = 30.0
    CLASS_WEIGHTS = [1.0,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     weight,
                     ]

BLOB_SIZE = 5
BLOB = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
])

CLASS_WEIGHTS_SUM = sum(CLASS_WEIGHTS)


VAL_DATA_SIZE = TRAIN_DATA_SIZE // 4
VAL_SUBSPLITS = 10
VALIDATION_STEPS = VAL_DATA_SIZE // BATCH_SIZE // VAL_SUBSPLITS
STEPS_PER_EPOCH = TRAIN_DATA_SIZE // BATCH_SIZE // 2




COLORS = np.array([
    [0,0,0],
    [255,255,255],
    [255,0,0],
    [0,255,0],
    [0,0,255],
    [138,43,226],
    [139,35,35],
    [152,245,255],
    [127,255,0],
    [69,139,0],
    [139,62,47],
    [0,238,238],
    [255,20,147],
    [255,215,0],
    [128,128,128],
    [255,211,155],
])