import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from unet import *
from dsb import DSB, Dataset


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


TRAIN_PATH = 'datasets/'

# Parameters

learning_rate = 1e-4
input_shape=(256,256,1)
loss = 'binary_crossentropy'
steps_per_epoch = 3500
epochs = 3
batch_size = 20
start_epoch = 2
last_step = 84
clear_all = False
prepare_dataset = False

dataset = DSB(TRAIN_PATH)

dataset.prepare()

# Setup optimizer

optimizer = keras.optimizers.Adam(lr=learning_rate)

# Build model

builder = Unet(input_shape=input_shape)
model = builder.build_model()
model.summary()

# Dataset generators

train_generator = dataset.generator('training', batch_size=batch_size)
valid_generator = dataset.generator('validation', batch_size=10)

# Metrics


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


metrics = ['accuracy', dice_coef]

# Compile model

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

v_generator = dataset.generator('validation', batch_size=1)
v_data = [next(v_generator) for item in range(5)]

# Train Model

model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                    initial_epoch=start_epoch,validation_data=valid_generator, validation_steps=30)

# Explore results


def generate_segmentation_mask(model, dataset, image):
    """
    Function generates segmentation mask for given image.
    Image willbe tiled. Tiles will be processed with the
    given model. Resulting tile masks will be combined
    in resulting mask.
    Parameters:
        model: trained keras model
        dataset: instance of DataScienceBowl2018 class
        image: numpy array that represent image to be segmented
    Return value: numpy array wit segmentation mask
    """
    mean = np.mean(image)
    if mean > 80:
        image = (255 - image)
    image_tiles = dataset.split_image_to_tiles(image)
    tile_num = len(image_tiles)
    tile_shape = image_tiles[0].shape
    batch = np.zeros((tile_num, tile_shape[0], tile_shape[1], 1))
    for tile_index in range(tile_num):
        tile = image_tiles[tile_index]
        tile = Dataset.normalize_image(tile)
        batch[tile_index, :, :, 0] = tile
    mask_batch = model.predict(batch)
    mask_tiles = []
    for tile_index in range(tile_num):
        tile = mask_batch[tile_index, :, :, 0]
        mask_tiles.append(tile)
    mask = dataset.combine_image_from_tiles(image.shape, mask_tiles)
    return mask


def show_test_prediction(number_of_test_images, error_threshold, show_diff=False):
    """
    Function shows images from the test set. For each image will be shown as well
    annotated segmentation mask, generated segmentation mask and difference between
    annotation and prediction.
    Parameters:
        number_of_test_images: number of test images to be shown
        error_threshold: show images only with error bigger as the threshold
        show_diff: if True, difference between prediction and annotation will be shown
    """
    t_generator = dataset.generator('test', batch_size=1)

    cols = 4
    rows = (number_of_test_images - 1) // cols + 1
    if cols > number_of_test_images:
        cols = number_of_test_images

    subrows = 3
    if show_diff:
        subrows = 4

    plt.figure(figsize=(5 * cols, 15 * rows))
    item = 0
    for image, label in t_generator:
        mask = generate_segmentation_mask(model, dataset, image)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        label1 = Dataset.normalize_label(label)
        pixel_dif = np.sum(np.abs(mask - label1)) / (image.shape[0] * image.shape[1] / 100)
        if pixel_dif >= error_threshold:
            row = item // cols
            col = item % cols
            plt.subplot2grid((rows * subrows, cols), (row * subrows, col))
            plt.xticks([])
            plt.yticks([])
            plt.title('error=' + str(pixel_dif))
            plt.imshow(image, cmap='gray')
            plt.subplot2grid((rows * subrows, cols), (row * subrows + 1, col))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(label1, cmap='gray')
            plt.subplot2grid((rows * subrows, cols), (row * subrows + 2, col))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(mask, cmap='gray')
            if show_diff:
                plt.subplot2grid((rows * subrows, cols), (row * subrows + 3, col))
                plt.xticks([])
                plt.yticks([])
                diff = np.abs(label1 - mask)
                plt.imshow(diff, cmap='gray')
            item += 1
        if item >= number_of_test_images:
            break
    plt.show()


# Show first 20 images from the test set.
# First row contains images, second - annotations, third - generated masks

show_test_prediction(20, 0)

# return dataset to initial

dataset.clear()