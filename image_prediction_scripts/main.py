import keras.layers

import usefull_blocks as blocks
import helpers as helpers


import tensorflow as tf
#create data
#print("TensorFlow version:", tf.__version__)
#mnist = tf.keras.datasets.mnist
#print("new Version")
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0

tf.random.set_seed(5);
x_train_d=tf.random.normal([32,    512,512,1], 0, 1, tf.float32, seed=1)
x_train_rgb=tf.random.normal([32,    512,512,3], 0, 1, tf.float32, seed=1)
x_train=[x_train_d,x_train_rgb]
#y_train=tf.random.normal([200,    128,128,1], 0, 1, tf.float32, seed=2)
#y_train=x_train
#x_test=tf.random.normal([100,       128,128,1], 0, 1, tf.float32, seed=3)
#x_test=x_train
#y_test=tf.random.normal([100,       128,128,1], 0, 1, tf.float32, seed=4)
#y_test=x_train

input_d = tf.keras.layers.Input(shape=(512, 512, 1))
input_rgb = tf.keras.layers.Input(shape=(512, 512, 3))


def get_d_branch(input,number_of_downsample_modules=3, number_of_non_downsample_modules=0,initial_channel_count=64):
    """
    creates a depth encoder branch, seems unnecesairily complicated, maybe just change back (compare rgb branch)
    :param input:
    :param number_of_downsample_modules:
    :param number_of_non_downsample_modules:
    :param initial_channel_count:
    :return: the branch
    """
    #TODO: exchange conv downsampling with advanced downsampling
    channel_count=initial_channel_count
    d_branch = tf.keras.layers.Conv2D(channel_count, 3,strides=(2,2), padding='same')(input) #128x128x64
    d_branch = tf.keras.layers.Conv2D(channel_count, 3,strides=(2,2), padding='same')(d_branch) #128x128x64
    d_branch = blocks.ResplacementBlock(channel_count)(d_branch)
    d_branch = blocks.ResplacementBlock(channel_count)(d_branch)
    channel_count=channel_count*2

    for i in range(0,number_of_downsample_modules-1):
        d_branch = keras.layers.MaxPooling2D()(d_branch)
        d_branch= blocks.ResplacementBlock(channel_count,apply_skip=False)(d_branch) #64x64x128
        d_branch= blocks.ResplacementBlock(channel_count)(d_branch)
        d_branch= blocks.ResplacementBlock(channel_count)(d_branch)
        channel_count = channel_count * 2

    return d_branch

def get_rgb_branch(input):
    rgb_branch= tf.keras.layers.Conv2D(64, 3,strides=(2,2), padding='same')(input)
    rgb_branch= tf.keras.layers.Conv2D(64, 3,strides=(2,2), padding='same')(rgb_branch)
    rgb_branch = blocks.ResplacementBlock(64) (rgb_branch)
    rgb_branch = blocks.ResplacementBlock(64) (rgb_branch)
    skip_output_1 = rgb_branch
    rgb_branch= keras.layers.MaxPooling2D()(rgb_branch)
    rgb_branch = blocks.ResplacementBlock(128,apply_skip=False) (rgb_branch)
    rgb_branch = blocks.ResplacementBlock(128) (rgb_branch)
    rgb_branch = blocks.ResplacementBlock(128) (rgb_branch)
    skip_output_2 = rgb_branch
    rgb_branch= keras.layers.MaxPooling2D()(rgb_branch)
    rgb_branch = blocks.ResplacementBlock(256,apply_skip=False) (rgb_branch)
    rgb_branch = blocks.ResplacementBlock(256) (rgb_branch)
    rgb_branch = blocks.ResplacementBlock(256) (rgb_branch)
    return rgb_branch , [skip_output_1,skip_output_2]


d_branch = get_d_branch(input_d)
rgb_branch,_ = get_rgb_branch(input_rgb)
merged = tf.keras.layers.Concatenate()([d_branch, rgb_branch])


model = tf.keras.models.Model(inputs=[input_d,input_rgb], outputs=merged)


model.compile(optimizer='adam',
              loss=helpers.segmentation_loss,
              metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=5)
#model.evaluate(x_test,  y_test, verbose=2)
#
#probability_model = tf.keras.Sequential([
#  model,
#  tf.keras.layers.Softmax()
#])

print(type(x_train))
print(x_train[0].shape)
print(len(x_train))
import time

print("starting a few runs")
for i in range(0,10):
    start=time.time()
    model.predict(x_train)
    stop=time.time()

    interference_time = stop - start

    print("interference took {} seconds for {} entries".format(interference_time,len(x_train[0])))
    print("thats {} ms per entry".format(1000*interference_time/len(x_train[0])))

print("end")

#probability_model(x_test[:5])
