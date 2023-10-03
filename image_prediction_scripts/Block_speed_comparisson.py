import usefull_blocks as blocks



import tensorflow as tf
#create data
#print("TensorFlow version:", tf.__version__)
#mnist = tf.keras.datasets.mnist
#print("new Version")
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0

tf.random.set_seed(5);
x_train=tf.random.normal([200,    128,128,3], 0, 3, tf.float32, seed=1)
y_train=tf.random.normal([200,    128,128,3], 0, 3, tf.float32, seed=2)
x_test=tf.random.normal([100,       128,128,3], 0, 3, tf.float32, seed=3)
y_test=tf.random.normal([100,       128,128,3], 0, 3, tf.float32, seed=4)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(256, 3, strides=(2, 2), padding='same'),
    #blocks.ResNetBlock(256),

    blocks.ResplacementBlock(256,local_filters_ratio=0.5,apply_attention=True,first_dilation_rate=2,second_dilation_rate=5),
    blocks.ResplacementBlock(256,local_filters_ratio=0.5,apply_attention=True,first_dilation_rate=2,second_dilation_rate=5),
    blocks.ResplacementBlock(256,local_filters_ratio=0.5,apply_attention=True,first_dilation_rate=2,second_dilation_rate=5),
    blocks.ResplacementBlock(256,local_filters_ratio=0.5,apply_attention=True,first_dilation_rate=2,second_dilation_rate=5),
    blocks.ResplacementBlock(256,local_filters_ratio=0.5,apply_attention=True,first_dilation_rate=2,second_dilation_rate=5),
    blocks.ResplacementBlock(256,local_filters_ratio=0.5,apply_attention=True,first_dilation_rate=2,second_dilation_rate=5),
    blocks.ResplacementBlock(256,local_filters_ratio=0.5,apply_attention=True,first_dilation_rate=2,second_dilation_rate=5),
    blocks.ResplacementBlock(256,local_filters_ratio=0.5,apply_attention=True,first_dilation_rate=2,second_dilation_rate=5),
])

print(x_train[:1].shape)

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




loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


model.compile(optimizer='adam',
              loss=segmentation_loss,
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
import time

print("starting a few runs")
for i in range(0,10):
    start=time.time()
    model.predict(x_train)
    stop=time.time()

    interference_time = stop - start

    print("interference took {} seconds for {} entries".format(interference_time,len(x_train)))
    print("thats {} ms per entry".format(1000*interference_time/len(x_train)))

print("end")

#probability_model(x_test[:5])
