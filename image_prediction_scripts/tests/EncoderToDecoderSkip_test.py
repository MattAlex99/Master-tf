
import tensorflow as tf
import helpers
import usefull_blocks
import usefull_blocks as ub
"""
This script is meant to test wether the execution of the given block works propperly
"""

# create some random inuts for testing
x_train =tf.random.normal([100,   64 ,64 ,128], 0, 3, tf.float32, seed=1)
y_train =tf.random.normal([100,   64 ,64 ,128], 0, 3, tf.float32, seed=2)
x_test =tf.random.normal([100,     64 ,64 ,128], 0, 3, tf.float32, seed=3)
y_test =tf.random.normal([100,     64 ,64 ,128], 0, 3, tf.float32, seed=4)

# define inputs
input_d = tf.keras.layers.Input(shape=(64 ,64, 128))
input_rgb = tf.keras.layers.Input(shape=(64 ,64 ,128))

# define how inputs aer processed
output = usefull_blocks.EncoderToDecoderSkip(32)
output=output([input_rgb, input_d])

# build and compile the model
model = tf.keras.models.Model(inputs=[input_d, input_rgb], outputs=output)
model.compile(optimizer='adam',
              loss=helpers.segmentation_loss,
              metrics=['accuracy'])

# run tests
import time

print("starting a few runs")
for i in range(0, 10):
    start = time.time()
    model.predict([x_train,y_train])
    stop = time.time()

    interference_time = stop - start

    print("interference took {} seconds for {} entries".format(interference_time, len(x_train[0])))
    print("thats {} ms per entry".format(1000 * interference_time / len(x_train[0])))

print("end")
