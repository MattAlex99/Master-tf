"""
This file serves to visualize how impactfull a shift in attentional weights is for a Neural Network
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf





def display_by_dense():
    x_size = (50, 50)
    x_len = 50 * 50
    x = np.ones(x_size, dtype=np.float32)


    mean_mult = 1.071
    std_derivation_mult = 0.253
    mean_bias = 0.114
    std_derivation_bias = 0.181

    #mean_mult = 1.0
    #std_derivation_mult = 0.1
    #mean_bias = 0.0
    #std_derivation_bias = 0.1


    x = x.flatten()
    x = np.expand_dims(x, 0)

    for i in range(0, 11):
        layer = tf.keras.layers.Dense(x_len,
                                      bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0))

        x = layer(x)
        x = tf.keras.layers.ReLU()(x)
        #x = tf.keras.activations.sigmoid(x)
        attention_mult = np.random.normal(loc=mean_mult, scale=std_derivation_mult, size=x.shape)
        attention_bias = np.random.normal(loc=mean_bias, scale=std_derivation_bias, size=x.shape)

        #x=x*attention_mult
        #x=x+attention_bias

        x = tf.keras.layers.ReLU()(x)

        print(x.shape)

        print("one done")
    # just apply weights
    print(np.mean(x))
    print(np.average(x))
    x = x.numpy()
    x[0, 0] = 0
    x[0, 1] = 4000
    x = np.reshape(x, x_size)

    x_for_display = np.clip(x, 0, 4000)
    plt.imshow(x_for_display)
    plt.show()



def display_by_CNN():
    x_size=(50,50)
    x_len= 50*50
    x=np.random.normal(loc=1, scale=0.5, size=x_size)
    print(x.shape)
    original=x
    mean_mult=1.071
    std_derivation_mult= 0.253

    mean_bias=0.114
    std_derivation_bias=0.181

    x=np.expand_dims(x,0)
    print(x.shape)
    x=np.expand_dims(x,-1)

    print(x.shape)
    num_of_filters=512
    for i in range(0,22):


        layer=tf.keras.layers.Conv2D(num_of_filters,3,padding="same",
                                    bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.2),
                                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.20,stddev=0.2))

        x=layer(x)
        x = x / (num_of_filters)
        x = tf.keras.activations.sigmoid(x)


        attention_mult=np.random.normal(loc=mean_mult, scale=std_derivation_mult, size=(1,1,1,num_of_filters))
        attention_bias=np.random.normal(loc=mean_bias, scale=std_derivation_bias, size=(1,1,1,num_of_filters))
        x=x*attention_mult
        x=x+attention_bias
        x = tf.keras.layers.ReLU()(x)

        print(x.shape)


    newX=np.zeros(x_size,dtype=np.float32)
    x=x.numpy()
    for i in range(0,num_of_filters):
        newX += x[:,:,:,i][0]
        print(np.average(x[:,:,:,i][0]))
    x=newX/num_of_filters
    #just apply weights
    print(np.median(x))
    print(np.average(x))
    x[0,0]=0.0
    x[0,1]=2.0
    x=np.reshape(x,x_size)

    x_for_display= np.clip(x,0,2)
    plt.imshow(x_for_display)
    plt.show()


if __name__ =="__main__":

    for i in range(0,100):
        display_by_dense()
    exit()
