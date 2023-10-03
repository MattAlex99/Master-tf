

import tensorflow as tf # note that tensorflow wont run on windows,
#please run in docker using docker run -it --rm -v "D:\Uni\Master3\pythonTestFolder":/tmp -w  /tmp tensorflow/tensorflow

#TODO: Fragen ob class based or function based layers are better

class SqueezeAndExcitationBlock(tf.keras.layers.Layer):
    """
    This blocks puropse is to be used as channel wide attention at multiple positions
    """
    def __init__(self, num_channels, first_fc_reduction_ratio=16):
        super(SqueezeAndExcitationBlock, self).__init__()
        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(int(num_channels/first_fc_reduction_ratio), activation='relu')
        self.fc2 = tf.keras.layers.Dense(num_channels, activation='sigmoid')

    def call(self, inputs):
        x = self.global_avg_pooling(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = tf.keras.activations.sigmoid(x)
        return x


class ResplacementBlock(tf.keras.layers.Layer):
    """
    Time requirement observations
     -  for higher channel counts increasingly faster then ResNet Block
     -  increasingly worse the higher percentage of operations is padding
        (worst case, small image, most operations semi local, high dilation rates)
    """
    def __init__(self, filters,local_filters_ratio=0.5,apply_attention=True,apply_skip=True, kernel_size=(3, 3), first_dilation_rate=3, second_dilation_rate=7, stride=(1, 1)):
        super(ResplacementBlock, self).__init__()
        self.apply_skip=apply_skip
        self.filters = filters
        self.kernel_size = kernel_size
        self.first_dilation_rate = first_dilation_rate
        self.second_dilation_rate = second_dilation_rate
        self.stride = stride
        self.apply_attention = apply_attention
        self.local_filters_ratio=local_filters_ratio #percentage of filters that will be used for local filters

    def build(self, input_shape):
        new_channel_count_x = int(self.filters * self.local_filters_ratio)
        new_channel_count_y = int(self.filters - new_channel_count_x)

        self.local_conv1 = tf.keras.layers.Conv2D(new_channel_count_x, [self.kernel_size[0], 1], strides=self.stride, padding='same')
        self.local_bn1 = tf.keras.layers.BatchNormalization()
        self.local_relu1 = tf.keras.layers.ReLU()

        self.local_conv2 = tf.keras.layers.Conv2D(new_channel_count_x, [1, self.kernel_size[1]], strides=self.stride, padding='same')
        self.local_bn2 = tf.keras.layers.BatchNormalization()
        self.local_relu2 = tf.keras.layers.ReLU()

        self.semi_local_conv1 = tf.keras.layers.Conv2D(new_channel_count_y, [self.kernel_size[0], 1], dilation_rate=self.first_dilation_rate, strides=self.stride, padding='same')
        self.semi_local_bn1 = tf.keras.layers.BatchNormalization()
        self.semi_local_relu1 = tf.keras.layers.ReLU()

        self.semi_local_conv2 = tf.keras.layers.Conv2D(new_channel_count_y, [1, self.kernel_size[1]], dilation_rate=self.first_dilation_rate, strides=self.stride, padding='same')
        self.semi_local_bn2 = tf.keras.layers.BatchNormalization()
        self.semi_local_relu2 = tf.keras.layers.ReLU()

        self.squeeze_excitation_x = SqueezeAndExcitationBlock(new_channel_count_x)
        self.squeeze_excitation_y = SqueezeAndExcitationBlock(new_channel_count_y)

    def call(self, inputs):
        shortcut = inputs
        x = inputs

        # Local channel
        x = self.local_conv1(x)
        x = self.local_bn1(x)
        x = self.local_relu1(x)

        x = self.local_conv2(x)
        x = self.local_bn2(x)
        x = self.local_relu2(x)

        if self.apply_attention:
            # Local channel attention
            local_attention_map = self.squeeze_excitation_x(x)

            # Element-wise multiplication
            x = tf.keras.layers.Multiply()([local_attention_map, x])

        y = inputs

        # Semi-local channel
        y = self.semi_local_conv1(y)
        y = self.semi_local_bn1(y)
        y = self.semi_local_relu1(y)

        y = self.semi_local_conv2(y)
        y = self.semi_local_bn2(y)
        y = self.semi_local_relu2(y)

        if self.apply_attention:
            # Semi-local channel attention
            semi_local_attention_map = self.squeeze_excitation_y(y)

            # Element-wise multiplication
            y = tf.keras.layers.Multiply()([semi_local_attention_map, y])

        # Concatenate local and semi-local features
        xy = tf.keras.layers.Concatenate(axis=-1)([x, y])
        if self.apply_skip:
            # Final addition
            output = tf.keras.layers.Add()([xy, shortcut])
        else:
            output=xy

        output = tf.keras.layers.ReLU()(output)

        return output


class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), stride=(1, 1), conv_shortcut=False):
        super(ResNetBlock, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_shortcut = conv_shortcut

    def build(self, input_shape):
        if self.conv_shortcut:
            self.shortcut_conv = tf.keras.layers.Conv2D(self.filters, (1, 1), strides=self.stride, padding='valid')
            self.shortcut_bn = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv2D(self.filters, self.kernel_size, strides=self.stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = inputs

        if self.conv_shortcut:
            shortcut = self.shortcut_conv(x)
            shortcut = self.shortcut_bn(shortcut)
        else:
            shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.ReLU()(x)

        return x