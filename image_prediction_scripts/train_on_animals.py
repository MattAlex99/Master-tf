import tensorflow as tf

import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

import usefull_blocks as blocks
from IPython.display import clear_output


dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
print(type(dataset))
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask
def load_image(datapoint):
  input_image = tf.image.resize(datapoint['image'], (256, 256))
  input_mask = tf.image.resize(
    datapoint['segmentation_mask'],
    (256, 256),
    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
  )

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 4
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

OUTPUT_CLASSES = 3
print(dataset['train'])
train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
print(train_images)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)


class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels


train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_images.batch(BATCH_SIZE)

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

for images, masks in train_batches.take(2):
  sample_image, sample_mask = images[0], masks[0]
  display([sample_image, sample_mask])



def get_initial_resplacement_module(filters,input):
    rgb_branch= tf.keras.layers.Conv2D(filters, 3,strides=(2,2), padding='same')(input)
    #rgb_branch= tf.keras.layers.Conv2D(filters, 3,strides=(2,2), padding='same')(rgb_branch)
    rgb_branch = blocks.ResplacementBlock(filters,local_filters_ratio=0.5,first_dilation_rate=1,second_dilation_rate=1) (rgb_branch)
    rgb_branch = blocks.ResplacementBlock(filters,local_filters_ratio=0.5,first_dilation_rate=1,second_dilation_rate=1) (rgb_branch)
    return rgb_branch

def get_regular_resplacement_module(filters, input,perform_pooling=True):
    if perform_pooling:
        input = tf.keras.layers.MaxPooling2D()(input)
    rgb_branch = blocks.ResplacementBlock(filters,local_filters_ratio=0.5,first_dilation_rate=3,second_dilation_rate=5,apply_skip=False) (input)
    rgb_branch = blocks.ResplacementBlock(filters,local_filters_ratio=0.5,first_dilation_rate=3,second_dilation_rate=5) (rgb_branch)
    rgb_branch = blocks.ResplacementBlock(filters,local_filters_ratio=0.5,first_dilation_rate=3,second_dilation_rate=5) (rgb_branch)
    return rgb_branch



"""
The section below will asemble a functional encoder 
"""

print("\n\nXXXXXXXX-Building model-XXXXXXXX\n\n")
#build encoder from individual blocks
input_d = tf.keras.layers.Input(shape=(512, 512, 1))
input_rgb = tf.keras.layers.Input(shape=(256, 256, 3))

#define encoder branches
rgb_branch= get_initial_resplacement_module(64,input_rgb) #128x128
rgb_skip_1=rgb_branch
rgb_branch= get_regular_resplacement_module(128,rgb_branch,perform_pooling=True) #64#64
rgb_skip_2=rgb_branch
rgb_branch= get_regular_resplacement_module(256,rgb_branch,perform_pooling=False) #32x32
#rgb_branch= get_regular_resplacement_module(256,rgb_branch,perform_pooling=False)
rgb_branch= get_regular_resplacement_module(512,rgb_branch,perform_pooling=False)

#depth_branch= get_initial_resplacement_module(64,input_d)
#depth_skip1=depth_branch
#depth_branch= get_regular_resplacement_module(128,depth_branch,perform_pooling=True)
#depth_skip2=depth_branch
#depth_branch= get_regular_resplacement_module(256,depth_branch,perform_pooling=True)
#depth_branch= get_regular_resplacement_module(256,depth_branch,perform_pooling=False)
#depth_branch= get_regular_resplacement_module(512,depth_branch,perform_pooling=False)

#rgb_branch,depth_branch=blocks.EncoderFusionBlock(512)([rgb_branch,depth_branch])
#merged = tf.keras.layers.Concatenate()([rgb_branch,depth_branch])
merged=rgb_branch

#define decoder
decoder =blocks.ResplacmentUpsamplingBlock(512)(merged) #64
decoder =blocks.ResplacmentUpsamplingBlock(256)(decoder)
#decoder =blocks.ResplacmentUpsamplingBlock(128)(decoder) #128
#decoder =blocks.ResplacmentUpsamplingBlock(64)(decoder) #256
#decoder =blocks.ResplacmentUpsamplingBlock(32)(decoder) # 512
decoder = tf.keras.layers.Conv2D(filters=OUTPUT_CLASSES,kernel_size=(3,3), activation='sigmoid',padding='same')(decoder)

model = tf.keras.models.Model(inputs=[input_rgb], outputs=decoder)

print("\n\nXXXXXXXX-Compiling model-XXXXXXXX\n\n")
gpus = tf.config.experimental.list_physical_devices('GPU')


model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'],
          )
model.summary()


def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]
def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print("\ndisplaying images\n")
    clear_output(wait=True)
    global test_batches
    show_predictions(test_batches)
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

class SaveModelCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print("\nsaving model")
    checkpoint_path="../../models_trained/model_{}"
    model.save_weights(checkpoint_path.format(epoch=epoch))
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))



EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_batches, epochs=EPOCHS,
                          steps_per_epoch= STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches,
                          callbacks=[DisplayCallback(),SaveModelCallback()])



def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]



show_predictions(test_batches, 3)
