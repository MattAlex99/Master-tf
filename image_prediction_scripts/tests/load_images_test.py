
import tensorflow as tf
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2


def direct_load_depth(depth_paths):
    result = []
    for path in depth_paths:
        depth = cv2.imread(path,
                           cv2.IMREAD_UNCHANGED)
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


    image_paths = [os.path.join(images_dir, filename) for filename in image_filenames]
    mask_paths = [os.path.join(masks_dir, filename) for filename in mask_filenames]
    depth_paths = [os.path.join(depth_dir, filename) for filename in depth_filenames]

    #TODO: hier ist noch ein workaround, weil sich exr dateien nicht dynamisch laden lassen. exr bilder werden sofort geladen,
    # dass solte noch geändert werden
    depth_images=direct_load_depth(depth_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths, depth_images))


    def load_and_preprocess_image(image_path, mask_path, depth_path):
        # Load and preprocess the image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, image_size) / 255
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.cast(image, tf.float32)

        # Load and preprocess the mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, image_size, method='nearest')
        mask = tf.cast(mask, tf.uint8)
        # mask = tf.keras.utils.to_categorical(mask, num_classes=num_classes)

        depth=depth_path
        # depth=  read_exr_to_tensor(depth_path)
        # print(depth_path.eval())
        #depth = tf.io.read_file(mask_path)
        #print(type(mask_path))
        #print(mask_path.numpy())
        #depth = cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)

        #print(depth)
        #print_image(depth)

        #depth=   tfio.experimental.image.decode_exr(depth,0,3,tf.float32)
        #depth = tf.image.decode_image(depth, channels=1, dtype=tf.float64)
        #depth=depth*100
        #print_image(depth)
        #depth = tf.convert_to_tensor(depth)
        # depth = tf.image.resize(depth, image_size, method='nearest')
        # depth = tf.cast(depth,tf.float32)
        return image, mask, depth

    # Map the load_and_preprocess_image function to each element in the dataset
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=len(image_paths)).batch(batch_size)

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
image, mask, depth = next(iterator)
# Display the image pairs
display_image_pairs(image, depth, mask, num_samples=5, figsize=(12, 6))
"""