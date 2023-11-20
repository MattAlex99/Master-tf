import tensorflow as tf
from tensorflow.python.framework import dtypes as _dtypes
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from os.path import isfile, join
def segmentation_loss(y_true, y_pred):
    """
    Compute the cross-entropy loss for segmentation maps.

    Args:
        y_true: True segmentation maps (ground truth), with shape (batch_size, height, width, num_classes).
        y_pred: Predicted segmentation maps, with the same shape as y_true.

    Returns:
        Cross-entropy loss.
    """
    # Flatten the true and predicted segmentation maps
    y_true_flat = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
    y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])

    # Compute the cross-entropy loss
    loss = tf.keras.losses.categorical_crossentropy(y_true_flat, y_pred_flat, from_logits=True)

    return loss




def read_regular_image(path):
    return cv2.imread(path)
def read_exr_image(path):
    return cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
def load_dataset_deprecated(base_path):
    """
    outdated because it loads all at once
    :param base_path: a path, that leads to the 'center folder' of the dataset including the three subfolders 'rgb','depth', 'mask'
    :return:
    """
    rgb_folder_path=base_path+"\\rgb\\"
    depth_folder_path=base_path+"\\depth\\"
    mask_folder_path=base_path+"\\mask\\"

    all_rgb_image_names = [f for f in os.listdir(rgb_folder_path) if os.path.isfile(os.path.join(rgb_folder_path, f))]
    all_image_names= [ i.split(".")[0] for i in all_rgb_image_names]

    rgb_images=[]
    depth_images=[]
    mask_images=[]

    for name in all_image_names:
        rgb_images.append(read_regular_image(rgb_folder_path+name+".png"))
        mask_images.append(read_regular_image(mask_folder_path+name+".png"))
        depth_images.append(read_exr_image(depth_folder_path+name+".exr"))
    return rgb_images,depth_images,mask_images


def direct_load_depth(depth_paths):
    result = []
    for path in depth_paths:
        depth = cv2.imread(path,
                           cv2.IMREAD_UNCHANGED)
        if len(np.shape(depth))>2:
            depth=cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        result.append(depth)
    return result
def load_segmentation_dataset(directory, image_size=(256, 256), num_classes=2, batch_size=32,
                              prefetch_buffer_size=tf.data.experimental.AUTOTUNE):
    """
  Load a segmentation dataset from a directory using tf.data.Dataset.

  Args:
      directory (str): The path to the dataset directory containing 'images' and 'masks' subdirectories.
      image_size (tuple): Target image size (height, width).
      num_classes (int): Number of segmentation classes.
      batch_size (int): Batch size for the dataset.
      prefetch_buffer_size (int): Number of batches to prefetch (tf.data.experimental.AUTOTUNE for dynamic prefetching).

  Returns:
      dataset (tf.data.Dataset): A tf.data.Dataset containing image-mask pairs.
  """
    images_dir = os.path.join(directory, 'rgb')
    masks_dir = os.path.join(directory, 'mask')
    depth_dir = os.path.join(directory, 'depth')

    image_filenames = os.listdir(images_dir)
    mask_filenames = os.listdir(masks_dir)
    depth_filenames = os.listdir(depth_dir)


    image_paths = [os.path.join(images_dir, filename) for filename in image_filenames] [0:-1:2]
    mask_paths = [os.path.join(masks_dir, filename) for filename in mask_filenames]    [0:-1:2]
    depth_paths = [os.path.join(depth_dir, filename) for filename in depth_filenames]  [0:-1:2]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, depth_paths,mask_paths))


    def load_and_preprocess_image(image_path,depth_path,mask_path):
        # Load and preprocess the image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, image_size) / 255
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.cast(image, tf.float32)
        channels = tf.unstack(image, axis=-1)
        image = tf.stack([channels[2], channels[1], channels[0]], axis=-1)

        # Load and preprocess the mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, image_size, method='nearest')
        mask = tf.cast(mask, tf.uint8)


        depth = tf.io.read_file(depth_path)
        depth = tf.image.decode_png(depth, channels=1,dtype=_dtypes.uint16)
        #depth_channels = tf.unstack(depth, axis=-1)

        depth= tf.image.resize(depth, image_size)
        depth= tf.cast(depth,tf.float32)
        #depth=tf.squeeze(depth,axis=-1)
        #depth= tf.expand_dims(depth,-1)

        return image, depth, mask

    # Map the load_and_preprocess_image function to each element in the dataset
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch the dataset
    if batch_size > 0:
        dataset = dataset.shuffle(buffer_size=len(image_paths)).batch(batch_size)
    else:
        dataset = dataset.shuffle(buffer_size=len(image_paths))

    # Prefetch for improved performance
    # dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    return dataset


import matplotlib.pyplot as plt
import numpy as np


def print_image(image):
    for e in image:
        for n in e:
            if n[0] > 0.0:
                print(n)


def display_image_pairs(images, depth, masks=None, num_samples=5, figsize=(12, 6)):
    """
    Display image pairs (input images and corresponding masks).

    Args:
        images (numpy.ndarray): An array of input images.
        masks (numpy.ndarray): An array of corresponding masks (optional).
        num_samples (int): Number of image pairs to display.
        figsize (tuple): Figure size (width, height).
    """
    # Ensure that the number of samples does not exceed the available data
    num_samples = min(num_samples, len(images))

    # Create subplots with 2 columns (input image and mask)
    fig, axes = plt.subplots(num_samples, 3, figsize=figsize)
    fig.subplots_adjust(hspace=0.3)

    for i in range(num_samples):
        # Display the input image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')

        print("depth")
        #print(depth[i])
        #print_image(depth[i])
        print("highest_ value:",np.max(depth[i]))
        axes[i, 1].imshow(depth[i])
        axes[i, 1].set_title('Input depth')
        axes[i, 1].axis('off')

        # Display the mask (if provided)
        if masks is not None:
            axes[i, 2].imshow(masks[i], cmap='viridis')  # Adjust the colormap as needed
            axes[i, 2].set_title('Mask')
            axes[i, 2].axis('off')

    plt.show()

"""
##usage of dataset creation
base_directory = "C:\\Uni\\Master\\tensorflow_ws\\Datasets\\processed\\cleargrasp"
train_dataset = load_segmentation_dataset(base_directory, image_size=(256, 256), num_classes=2,
                                          batch_size=32)
#usage of display dataset
iterator = iter(train_dataset)
image, depth, mask = next(iterator)
# Display the image pairs
display_image_pairs(image, depth, mask, num_samples=5, figsize=(12, 6))
"""


