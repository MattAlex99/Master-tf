import keras.models
import tensorflow as tf

#import tensorflow_datasets as tfds
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import matplotlib.pyplot as plt
import numpy as np
import helpers
import usefull_blocks
import usefull_blocks as blocks
from IPython.display import clear_output


img_height = 256
img_width = 256
BATCH_SIZE = 4
TRAIN_SPLIT=0.8
BUFFER_SIZE = 2
OUTPUT_CLASSES = 2
CONTINUE_TRAINING=False
model_path="../../models_trained/architecture_reference/ResNet_Based_rgb_only"
#base_directory = "../../Datasets/grasp_point_detection/ok"
base_directory = "../../Datasets/reference/cleargrasp/"

dataset_directory = base_directory

text_dataset= helpers.get_path_names_dataset(base_directory)
dataset= text_dataset.map(helpers.map_name_dataset)


reduce_size=0
print("\nPreparing dataset\nThis may take a while")
#dataset  = helpers.load_segmentation_dataset(dataset_directory,
#                                                  image_size=(img_height, img_width),
#                                                  num_classes=OUTPUT_CLASSES,
#                                                  batch_size=-1)
print("\ninitial dataset loading complete\n")

TRAIN_LENGTH = int((len(dataset)-reduce_size)  * TRAIN_SPLIT)
VAL_LENGTH= int((len(dataset)-reduce_size  )* (1-TRAIN_SPLIT) )
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE


dataset=dataset.skip(reduce_size)

train_images=dataset.take(TRAIN_LENGTH)
test_images= dataset.skip(TRAIN_LENGTH)






class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_flip_rgb = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_flip_depth = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_flip_mask = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    self.augment_rotation_rgb= tf.keras.layers.RandomRotation( [-0.2,0.2],seed=seed)
    self.augment_rotation_depth= tf.keras.layers.RandomRotation( [-0.2,0.2],seed=seed)
    self.augment_rotation_mask= tf.keras.layers.RandomRotation( [-0.2,0.2],seed=seed)

    self.augment_brightness=tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0.0,1.0), seed=seed)

    height_factor=(-0.2,0.2)
    width_factor=(-0.2,0.2)
    self.augment_zoom_rgb=   tf.keras.layers.RandomZoom(height_factor=height_factor, width_factor=width_factor,seed=seed)
    self.augment_zoom_depth= tf.keras.layers.RandomZoom(height_factor=height_factor, width_factor=width_factor,seed=seed)
    self.augment_zoom_mask=  tf.keras.layers.RandomZoom(height_factor=height_factor, width_factor=width_factor,seed=seed)

  def call(self, inputs, labels):
    inputs_rgb=inputs[...,0:3]
    inputs_d=inputs[...,-1]
    inputs_d=tf.expand_dims(inputs_d,-1)
    inputs_rgb = self.augment_flip_rgb(inputs_rgb)
    inputs_d =   self.augment_flip_depth(inputs_d)
    labels =     self.augment_flip_mask(labels)

    inputs_rgb = self.augment_rotation_rgb(inputs_rgb)
    inputs_d =   self.augment_rotation_depth(inputs_d)
    labels =     self.augment_rotation_mask(labels)

    inputs_rgb = self.augment_brightness(inputs_rgb)

    inputs_rgb = self.augment_zoom_rgb(inputs_rgb)
    inputs_d =   self.augment_zoom_depth(inputs_d)
    labels =     self.augment_zoom_mask(labels)

    inputs= tf.concat([inputs_rgb,inputs_d], axis=-1)
    return inputs, labels

def make_rgbd_tensor(rgb_tensor, depth_tensor, mask_tensor):
    # Concatenate the RGB and depth tensors along the last axis to create the input tensor.
    input_tensor = tf.concat([rgb_tensor, depth_tensor/1000], axis=-1)
    return input_tensor, mask_tensor/255


train_batches = (
    train_images
    #.cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    #.cache()
    #.map(make_rgbd_tensor)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

print("train batches complete")
test_batches = (
    test_images
    #.cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    #.map(make_rgbd_tensor)
    .repeat()
    .prefetch(buffer_size=tf.data.AUTOTUNE)#
)
print("batches created")
print(type(test_batches))

#for i in range (5):
#    for batch in train_batches.take(5):
#        print(batch[0].shape)
#        print(batch[1].shape)
#        rgbd=batch[0][0]
#        mask=np.array(batch[1][0])
#        rgb=np.array(rgbd[...,0:3])
#        depth=np.array(rgbd[...,-1])
#        print(np.shape(rgbd))
#        print(np.shape(rgb))
#        print(np.shape(depth))
#        cv2.imshow("depth",depth)
#        cv2.imshow("rgb",rgb)
#        cv2.imshow("mask",mask*255)
#        cv2.waitKey(0)
#exit()

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image','depth', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

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
#input_d = tf.keras.layers.Input(shape=(256, 256, 1))
input_rgbd = tf.keras.layers.Input(shape=(256, 256, 4))
input_rgb,input_d=usefull_blocks.ChannelSplitter()(input_rgbd)
#input_d=input_rgb
#define encoder branches
rgb_branch= get_initial_resplacement_module(64,input_rgb) #128x128
rgb_skip_1=rgb_branch
rgb_branch= get_regular_resplacement_module(128,rgb_branch,perform_pooling=True) #64#64
rgb_skip_2=rgb_branch
rgb_branch= get_regular_resplacement_module(256,rgb_branch,perform_pooling=False) #32x32
#rgb_branch= get_regular_resplacement_module(256,rgb_branch,perform_pooling=False)
rgb_branch= get_regular_resplacement_module(512,rgb_branch,perform_pooling=False)

depth_branch= get_initial_resplacement_module(64,input_d)
depth_skip1=depth_branch
depth_branch= get_regular_resplacement_module(128,depth_branch,perform_pooling=True)
depth_skip2=depth_branch
depth_branch= get_regular_resplacement_module(256,depth_branch,perform_pooling=False)
#depth_branch= get_regular_resplacement_module(256,depth_branch,perform_pooling=False)
depth_branch= get_regular_resplacement_module(512,depth_branch,perform_pooling=False)

#rgb_branch,depth_branch=blocks.EncoderFusionBlock(512)([rgb_branch,depth_branch])
merged = tf.keras.layers.Concatenate()([rgb_branch,depth_branch])
#merged=rgb_branch

#define decoder
#decoder =blocks.ResplacmentUpsamplingBlock(512)(merged) #64
decoder =blocks.ResplacmentUpsamplingBlock(256)(merged) #64
decoder =blocks.ResplacmentUpsamplingBlock(128)(decoder)
#decoder =blocks.ResplacmentUpsamplingBlock(128)(decoder) #128
#decoder =blocks.ResplacmentUpsamplingBlock(64)(decoder) #256
#decoder =blocks.ResplacmentUpsamplingBlock(32)(decoder) # 512
decoder = tf.keras.layers.Conv2D(filters=OUTPUT_CLASSES,kernel_size=(3,3),padding='same')(decoder)
#decoder = tf.keras.layers.Activation("sigmoid")(decoder)
#decoder = tf.keras.layers.ReLU()(decoder)
decoder=  tf.keras.layers.Softmax()(decoder)

#model = tf.keras.models.Model(inputs=input_rgbd, outputs=decoder)

model = blocks.get_ResNet_model(2)
print("\n\nXXXXXXXX-Compiling model-XXXXXXXX\n\n")
gpus = tf.config.experimental.list_physical_devices('GPU')


model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
          metrics=['accuracy'],
          )
model.summary()


def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]
def show_predictions(dataset=None, num=1):
  if dataset:
    for input, mask in dataset.take(num):
      pred_mask = model.predict(input)
      image=input[0][...,0:3]
      depth=tf.expand_dims(input[0][...,3],-1)
      display([image,depth, mask[0], create_mask(pred_mask)])
  else:
    pass
    #display([sample_image, sample_mask,
     #        create_mask(model.predict(sample_image[tf.newaxis, ...]))])


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print("\nmaking prediction\n")
    clear_output(wait=True)
    global test_batches
    show_predictions(test_batches,num=3)
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

class SaveImageCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print("\nmaking prediction\n")
    global test_batches
    i=0
    for input, mask in test_batches.take(3):
        pred_mask = model.predict(input)
        image = input[0][..., 0:3]*255

        depth = tf.expand_dims(input[0][..., 3], -1)

        cv2.imwrite("../images/eopch_{}_result_{}_rgb.png".format(epoch,i),image.numpy())
        cv2.imwrite("../images/eopch_{}_result_{}_label.png".format(epoch,i),mask[0].numpy()*100)
        cv2.imwrite("../images/eopch_{}_result_{}_prediction.png".format(epoch,i),create_mask(pred_mask).numpy()*(255/(OUTPUT_CLASSES)))
        i+=1
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

class SaveModelCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if epoch % 4 ==1:
        print("\nsaving model\n")
        global model_path
        model.save(model_path)
        print ('\nmodel saved epoch {}\n'.format(str(epoch+1)))


EPOCHS = 25
VAL_SUBSPLITS = 5
VALIDATION_STEPS = int (VAL_LENGTH)


#model.save(model_path)
#del model
#
#model=keras.models.load_model(model_path)

if CONTINUE_TRAINING:
#if False:
    model = keras.models.load_model(model_path)


#for input, mask in test_batches.take(3):
#    input=input.numpy()[0]
#    mask=mask.numpy()[0]
#    print(np.shape(input))
#    depth= input[...,-1]
#    rgb= input[...,0:-1]
#    for i in mask:
#        print(i)
#    cv2.imshow("depth",depth)
#    cv2.imshow("mask",mask)
#    cv2.imshow("rgb",rgb)
#    cv2.waitKey(0)




history = model.fit(train_batches,
                          epochs= EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches,
                          callbacks=[SaveModelCallback()])



print("final model save")
model.save(model_path)


print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
