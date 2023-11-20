import keras.layers

import usefull_blocks as blocks
import helpers as helpers
import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import os
print(os.listdir("./images/")) # returns list




tf.random.set_seed(5);
#x_train_d=tf.random.normal([16,    512,512,1], 0, 1, tf.float32, seed=1)
#x_train_rgb=tf.random.normal([16,    512,512,3], 0, 1, tf.float32, seed=1)
#x_train=[x_train_rgb,x_train_d]

relative_img_folder_path="./images/"

train_label=  np.float32(cv2.imread(relative_img_folder_path+"label_123.png",cv2.IMREAD_GRAYSCALE) /255)
train_label = np.expand_dims(train_label,-1)

x_train_label=tf.convert_to_tensor([train_label,train_label,train_label]*1)
train_image_1=np.float32(cv2.imread(relative_img_folder_path+"img_1.png") /255)
train_image_2=np.float32(cv2.imread(relative_img_folder_path+"img_2.png") /255)
train_image_3=np.float32(cv2.imread(relative_img_folder_path+"img_3.png") /255)
x_train_img = tf.convert_to_tensor([train_image_1,train_image_2,train_image_3]*1,dtype=tf.float32)
x_train=[x_train_img,x_train_label]

#for line in train_image_2[80]:
#    print(line)
#cv2.imshow('image',train_image_2)
#cv2.waitKey(0)
#
x_train_val=x_train_label


for image in x_train_val:
    #print(image)
    imgplot = plt.imshow(image)
    plt.show()


def get_initial_resplacement_module(filters,input):
    rgb_branch= tf.keras.layers.Conv2D(filters, 3,strides=(2,2), padding='same')(input)
    #rgb_branch= tf.keras.layers.Conv2D(filters, 3,strides=(2,2), padding='same')(rgb_branch)
    rgb_branch = blocks.ResplacementBlock(filters,local_filters_ratio=0.5,first_dilation_rate=1,second_dilation_rate=1) (rgb_branch)
    rgb_branch = blocks.ResplacementBlock(filters,local_filters_ratio=0.5,first_dilation_rate=1,second_dilation_rate=1) (rgb_branch)
    return rgb_branch

def get_regular_resplacement_module(filters, input,perform_pooling=True):
    if perform_pooling:
        input = keras.layers.MaxPooling2D()(input)
    rgb_branch = blocks.ResplacementBlock(filters,local_filters_ratio=0.5,first_dilation_rate=1,second_dilation_rate=1,apply_skip=False) (input)
    rgb_branch = blocks.ResplacementBlock(filters,local_filters_ratio=0.5,first_dilation_rate=1,second_dilation_rate=1) (rgb_branch)
    rgb_branch = blocks.ResplacementBlock(filters,local_filters_ratio=0.5,first_dilation_rate=1,second_dilation_rate=1) (rgb_branch)
    return rgb_branch



"""
The section below will asemble a functional encoder 
"""

print("\n\nXXXXXXXX-Building model-XXXXXXXX\n\n")
#build encoder from individual blocks
input_d = tf.keras.layers.Input(shape=(512, 512, 1))
input_rgb = tf.keras.layers.Input(shape=(512, 512, 3))

#define encoder branches
rgb_branch= get_initial_resplacement_module(64,input_rgb) #128x128
rgb_skip_1=rgb_branch
rgb_branch= get_regular_resplacement_module(128,rgb_branch,perform_pooling=True) #64#64
rgb_skip_2=rgb_branch
rgb_branch= get_regular_resplacement_module(256,rgb_branch,perform_pooling=True) #32x32
rgb_branch= get_regular_resplacement_module(256,rgb_branch,perform_pooling=False)
rgb_branch= get_regular_resplacement_module(512,rgb_branch,perform_pooling=False)

depth_branch= get_initial_resplacement_module(64,input_d)
depth_skip1=depth_branch
depth_branch= get_regular_resplacement_module(128,depth_branch,perform_pooling=True)
depth_skip2=depth_branch
depth_branch= get_regular_resplacement_module(256,depth_branch,perform_pooling=True)
depth_branch= get_regular_resplacement_module(256,depth_branch,perform_pooling=False)
depth_branch= get_regular_resplacement_module(512,depth_branch,perform_pooling=False)

rgb_branch,depth_branch=blocks.EncoderFusionBlock(512)([rgb_branch,depth_branch])
merged = tf.keras.layers.Concatenate()([rgb_branch,depth_branch])


#define decoder
decoder =blocks.ResplacmentUpsamplingBlock(512)(merged) #64
decoder =blocks.ResplacmentUpsamplingBlock(256)(decoder)
decoder =blocks.ResplacmentUpsamplingBlock(128)(decoder) #128
#decoder =blocks.ResplacmentUpsamplingBlock(64)(decoder) #256
#decoder =blocks.ResplacmentUpsamplingBlock(32)(decoder) # 512
decoder = tf.keras.layers.Conv2D(filters=1,kernel_size=(3,3), activation='sigmoid',padding='same')(decoder)

model = tf.keras.models.Model(inputs=[input_rgb,input_d], outputs=decoder)

print("\n\nXXXXXXXX-Compiling model-XXXXXXXX\n\n")
gpus = tf.config.experimental.list_physical_devices('GPU')


model.compile(optimizer='adam',
          loss=tf.keras.losses.binary_crossentropy,
          metrics=['accuracy'],
          )
model.summary()

model.fit(x_train, x_train_val, epochs=50,batch_size=4)
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
    result_tensor=model.predict(x_train)
    stop=time.time()

    interference_time = stop - start
    print(tf.shape(result_tensor))
    print("interference took {} seconds for {} entries".format(interference_time,len(x_train[0])))
    print("thats {} ms per entry".format(1000*interference_time/len(x_train[0])))

    for image in result_tensor:
        #print(image)
        imgplot = plt.imshow(image)
        plt.show()

print("end")

#probability_model(x_test[:5])
