import os

import cv2
import h5py
import numpy as np
import shutil

from matplotlib import pyplot as plt
import tensorflow as tf

from constants import START_DIR, IMAGES_DIR, MASKS_DIR, IMAGES_SIZE1, \
    IMAGES_SIZE2, CLASSES, BLOB, BLOB_SIZE, OG_SIZE1, OG_SIZE2
from util import display_images

global image_counter
image_counter = 1

def read_h5(path):
    f = h5py.File(path+'CollectedData_tinsttk.h5', 'r')
    # print(list(f.keys()))

    test = f['keypoints']
    # print(test)

    labels = None
    positions = None

    image_names = None
    for hmm in test.keys():
        # print(test[hmm], test[hmm].shape)
        # print(test[hmm][:])
        # print()

        if hmm == 'block0_items_level1':
            labels = test[hmm][:]

        if hmm == 'block0_values':
            positions = test[hmm][:]

        if hmm == 'axis1_level2':
            image_names = test[hmm][:]

    # print(labels)

    coords = np.zeros((len(positions), len(labels), 2), dtype=object)
    for id, (photos, image_name) in enumerate(zip(positions, image_names)):
        coords[id] = np.reshape(photos, (len(labels), 2))
        # print(image_name)
        # print(coords[id].shape)

    return image_names, coords





def save_images_and_masks(path, image_names, coordinates):
    global image_counter
    for image_name, coordinate in zip(image_names, coordinates):
        rs_coordinates = np.array(coordinate).reshape((-1, 2))
        bodypart_masks = np.zeros((IMAGES_SIZE1, IMAGES_SIZE2, CLASSES))
        for id, center in enumerate(rs_coordinates):
            if np.isnan(center[0]) or np.isnan(center[1]):
                continue
            mask = np.zeros((IMAGES_SIZE1, IMAGES_SIZE2))  # keep it 1 channel
            # mask[int(center[1]), int(center[0])] = 255
            # mask = cv2.GaussianBlur(mask, (7-id*2, 7-id*2), 0)
            # mask = cv2.GaussianBlur(mask, (5, 5), 0)
            # mask_max = np.amax(mask)
            # mask = mask / mask_max * 255
            # # no gaussian start
            # bool_mask = mask > 35  # for k5
            # # bool_mask = mask > 25 # for k9
            # int_mask = np.array(bool_mask, dtype=np.int)
            # mask = int_mask * (id + 1)


            x = int(center[1] / (OG_SIZE1 * IMAGES_SIZE1))
            y = int(center[0] / (OG_SIZE2 * IMAGES_SIZE2))
            mask[x, y] = 255
            mask[x - BLOB_SIZE//2:x+BLOB_SIZE//2+1, y-BLOB_SIZE//2:y+BLOB_SIZE//2+1] = 255
            mask[x - BLOB_SIZE // 2:x + BLOB_SIZE // 2 + 1, y - BLOB_SIZE // 2:y + BLOB_SIZE // 2 + 1] *= BLOB
            bool_mask = mask == 255  # for k5
            int_mask = np.array(bool_mask, dtype=np.int)
            mask = int_mask * (id + 1)


            bodypart_masks[:, :, id] = mask

        final_mask = np.zeros((IMAGES_SIZE1, IMAGES_SIZE2), dtype=int)
        for i in range(final_mask.shape[0]):
            for j in range(final_mask.shape[1]):
                if np.count_nonzero(bodypart_masks[i, j]) == 1:
                    final_mask[i, j] = np.sum(bodypart_masks[i, j])


        image_name = image_name.decode('utf-8')
        image = cv2.imread(path + image_name)
        image = cv2.resize(image, (IMAGES_SIZE1, IMAGES_SIZE2), interpolation=cv2.INTER_LINEAR)

        if np.any(final_mask > 3) == True:
            final_mask = final_mask[:, :, None]
            print(np.unique(final_mask), image.shape, final_mask.shape)
            display_images([image, final_mask])

        new_name = f"img{image_counter}.png"
        cv2.imwrite(IMAGES_DIR + f"{new_name}", np.uint8(image))
        cv2.imwrite(MASKS_DIR + f"mask_{new_name}", np.uint8(final_mask))
        image_counter+=1


for data_folder in os.listdir(START_DIR):
    image_names, coords = read_h5(START_DIR+data_folder+"/")

    # print(image_names[0].decode('utf-8'))
    # new_names =  # TODO, verify if they are not overwritten
    # for id, image_name in enumerate(image_names):
    #     shutil.copyfile(START_DIR+data_folder+"/"+image_name.decode('utf-8'), IMAGES_DIR+"/"+image_name.decode('utf-8'))

    # save_images_and_masks(START_DIR+data_folder+"/", image_names, coords)
    save_images_and_masks(START_DIR+data_folder+"/", image_names, coords)

