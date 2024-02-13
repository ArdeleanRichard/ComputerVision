import matplotlib.pyplot as plt
import tensorflow as tf

from constants import MODEL_DIR, MODEL_NAME


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def display_images(display_list, save=False):
    plt.figure(figsize=(15, 6))

    if len(display_list) == 3:
        title = ['Input Image', 'True Mask', 'Predicted Mask']
    elif len(display_list) == 2:
        title = ['Input Image', 'True Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='jet', interpolation="none")
        plt.axis('off')
    if save:
        plt.savefig(MODEL_DIR + '/' + MODEL_NAME + "_show" + '.png')
    plt.show()


def show_predictions_on_data(model, sample_image, sample_mask):
    display_images([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])



def plot_loss_n_acc(loss, acc, val_loss, val_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.plot(acc, label='accuracy')
    ax1.plot(val_acc, label='validation accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0.0, 1])
    ax1.legend(loc='lower right')
    ax2.plot(loss, label='loss')
    ax2.plot(val_loss, label='validation loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='lower right')
    plt.savefig(MODEL_DIR + '/' + MODEL_NAME + "_acc_loss" + '.png')
    plt.show()


def plot_history(model_history):
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    plot_loss_n_acc(loss, acc, val_loss, val_acc)