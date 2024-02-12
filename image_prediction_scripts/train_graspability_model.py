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
model_path="../../models_trained/architecture_graspabiltiy/test_more_local_non_normalized_max_distance800_no_add_attention_to_rgb"
#base_directory = "../../Datasets/grasp_point_detection/ok"
base_directory = "../../Datasets/Datasets_grip_orientation/003"

dataset_directory = base_directory

text_dataset= helpers.get_path_names_orientation_datset(base_directory)

dataset= text_dataset.map(helpers.map_name_angle_dataset)


def display_from_dataset(dataset,count):
    for i in dataset.take(count).as_numpy_iterator():
        data=i
        rgbd_batch=data[0]
        rgb=rgbd_batch[0][...,0:3]
        depth=rgbd_batch[0][...,3]
        label_batch=data[1]

        mask_graspability=label_batch[0][0]
        mask_horzontal=np.abs(label_batch[1][0]) #negative values would be dispayed as
        mask_vertical= np.abs(label_batch[2][0])
        cv2.imshow("rgb",rgb)
        cv2.imshow("depth",depth)
        cv2.imshow("mask_graspability",mask_graspability)
        cv2.imshow("mask_horizontal",mask_horzontal)
        cv2.imshow("mask_vertical",mask_vertical)
        #for n in mask_horzontal:
        #    print(n)
        plt.imshow(depth)
        #plt.axis('off')
        plt.show()
        cv2.waitKey(0)

TRAIN_LENGTH = int(len(dataset)  * TRAIN_SPLIT)
VAL_LENGTH= int(len(dataset)  * (1-TRAIN_SPLIT) )
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE


train_images=dataset.take(TRAIN_LENGTH)
test_images= dataset.skip(TRAIN_LENGTH)


class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    #TODO Alter augment in a way that makes it work for this kind of data

    #both use the same seed, so they'll make the same random changes.
    #self.augment_flip_rgb = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    #self.augment_flip_depth = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    #self.augment_flip_mask = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    #self.augment_rotation_rgb= tf.keras.layers.RandomRotation( [-0.2,0.2],seed=seed)
    #self.augment_rotation_depth= tf.keras.layers.RandomRotation( [-0.2,0.2],seed=seed)
    #self.augment_rotation_mask= tf.keras.layers.RandomRotation( [-0.2,0.2],seed=seed)

    self.augment_brightness=tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0.0,1.0), seed=seed)

    height_factor=(-0.4,0.4)
    width_factor=(-0.4,0.4)
    self.augment_zoom_rgb=          tf.keras.layers.RandomZoom(fill_mode="constant",fill_value=0.0,interpolation="bilinear",height_factor=height_factor, width_factor=width_factor,seed=seed)
    self.augment_zoom_depth=        tf.keras.layers.RandomZoom(fill_mode="constant",fill_value=0.0,interpolation="bilinear",height_factor=height_factor, width_factor=width_factor,seed=seed)
    self.augment_zoom_mask_horiz=   tf.keras.layers.RandomZoom(fill_mode="constant",fill_value=0.0,interpolation="nearest",height_factor=height_factor, width_factor=width_factor,seed=seed)
    self.augment_zoom_mask_grip=    tf.keras.layers.RandomZoom(fill_mode="constant",fill_value=0.0,interpolation="nearest",height_factor=height_factor, width_factor=width_factor,seed=seed)
    self.augment_zoom_mask_verti=   tf.keras.layers.RandomZoom(fill_mode="constant",fill_value=0.0,interpolation="nearest",height_factor=height_factor, width_factor=width_factor,seed=seed)

  def call(self, inputs, labels):
    inputs_rgb=inputs[...,0:3]
    inputs_d=inputs[...,-1]
    inputs_d=tf.expand_dims(inputs_d,-1)
    grip,horiz,verti=labels
    #inputs_rgb = self.augment_flip_rgb(inputs_rgb)
    #inputs_d =   self.augment_flip_depth(inputs_d)
    #labels =     self.augment_flip_mask(labels)

    #inputs_rgb = self.augment_rotation_rgb(inputs_rgb)
    #inputs_d =   self.augment_rotation_depth(inputs_d)
    #labels =     self.augment_rotation_mask(labels)

    inputs_rgb = self.augment_brightness(inputs_rgb)

    inputs_rgb = self.augment_zoom_rgb(inputs_rgb)
    inputs_d =   self.augment_zoom_depth(inputs_d)
    horiz =      self.augment_zoom_mask_horiz(horiz)
    grip =       self.augment_zoom_mask_grip(grip)
    verti =      self.augment_zoom_mask_verti(verti)

    inputs= tf.concat([inputs_rgb,inputs_d], axis=-1)
    return inputs, (grip,horiz,verti)




train_batches = (
    train_images
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
#display_from_dataset(train_batches,100)

print("train batches complete")
test_batches = (
    test_images
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .prefetch(buffer_size=tf.data.AUTOTUNE)#
)
print("batches created")
print(type(test_batches))


def get_initial_resplacement_module(filters,input):
    rgb_branch= tf.keras.layers.Conv2D(filters, 3,strides=(2,2), padding='same')(input)
    #rgb_branch= tf.keras.layers.Conv2D(filters, 3,strides=(2,2), padding='same')(rgb_branch)
    rgb_branch = blocks.ResplacementBlock(filters,local_filters_ratio=0.8,first_dilation_rate=1,second_dilation_rate=1) (rgb_branch)
    rgb_branch = blocks.ResplacementBlock(filters,local_filters_ratio=0.8,first_dilation_rate=1,second_dilation_rate=1) (rgb_branch)
    return rgb_branch

def get_regular_resplacement_module(filters, input,perform_pooling=True):
    if perform_pooling:
        input = tf.keras.layers.MaxPooling2D()(input)
    rgb_branch = blocks.ResplacementBlock(filters,local_filters_ratio=0.8,first_dilation_rate=3,second_dilation_rate=5,apply_skip=False) (input)
    rgb_branch = blocks.ResplacementBlock(filters,local_filters_ratio=0.8,first_dilation_rate=3,second_dilation_rate=5) (rgb_branch)
    rgb_branch = blocks.ResplacementBlock(filters,local_filters_ratio=0.8,first_dilation_rate=3,second_dilation_rate=5) (rgb_branch)
    return rgb_branch



"""
The section below will asemble a functional encoder 
"""


print("\n\nXXXXXXXX-Building model-XXXXXXXX\n\n")
#build encoder from individual blocks
input_rgbd = tf.keras.layers.Input(shape=(256, 256, 4))
input_rgb,input_d=usefull_blocks.ChannelSplitter()(input_rgbd)
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

rgb_branch,depth_branch=blocks.EncoderFusionBlock(512)([rgb_branch,depth_branch])
merged = tf.keras.layers.Concatenate()([rgb_branch,depth_branch])
#merged=rgb_branch

#define decoder
#decoder =blocks.ResplacmentUpsamplingBlock(512)(merged) #64
decoder =blocks.ResplacmentUpsamplingBlock(256)(merged) #64
decoder =blocks.ResplacmentUpsamplingBlock(128)(decoder)

decoder_angle_horizontal = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same')(decoder)
decoder_angle_horizontal=tf.keras.layers.BatchNormalization()(decoder_angle_horizontal)
decoder_angle_horizontal = tf.keras.layers.ReLU()(decoder_angle_horizontal)
decoder_angle_horizontal = tf.keras.layers.Conv2D(filters=1,kernel_size=(3,3),padding='same')(decoder_angle_horizontal)
decoder_angle_horizontal=  tf.keras.layers.Activation('tanh',name="horizontal_out")(decoder_angle_horizontal)


decoder_angle_vertical = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same')(decoder)
decoder_angle_vertical=tf.keras.layers.BatchNormalization()(decoder_angle_vertical)
decoder_angle_vertical = tf.keras.layers.ReLU()(decoder_angle_vertical)
decoder_angle_vertical = tf.keras.layers.Conv2D(filters=1,kernel_size=(3,3),padding='same')(decoder_angle_vertical)
decoder_angle_vertical=  tf.keras.layers.Activation('tanh',name="vertical_out")(decoder_angle_vertical)

decoder_gripability = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same')(decoder)
decoder_gripability=tf.keras.layers.BatchNormalization()(decoder_gripability)
decoder_gripability = tf.keras.layers.ReLU()(decoder_gripability)
decoder_gripability = tf.keras.layers.Conv2D(filters=2,kernel_size=(3,3),padding='same')(decoder_gripability)
decoder_gripability=  tf.keras.layers.Softmax(name='grip_out')(decoder_gripability)

#output=tf.keras.layers.Concatenate()([decoder_gripability,decoder_angle_horizontal,decoder_angle_vertical])
model = tf.keras.models.Model(inputs=input_rgbd, outputs=[decoder_gripability,decoder_angle_horizontal,decoder_angle_vertical])
#model = tf.keras.models.Model(inputs=input_rgbd, outputs=output)

print("\n\nXXXXXXXX-Compiling model-XXXXXXXX\n\n")
gpus = tf.config.experimental.list_physical_devices('GPU')


model.compile(optimizer='adam',
          loss={'horizontal_out':tf.keras.losses.MeanSquaredError(),
                'vertical_out':tf.keras.losses.MeanSquaredError(),
                'grip_out':tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                },
          metrics=['accuracy'],
          )
model.summary()

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




if CONTINUE_TRAINING:
    model = keras.models.load_model(model_path)

history = model.fit(train_batches,
                          verbose=2,
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
