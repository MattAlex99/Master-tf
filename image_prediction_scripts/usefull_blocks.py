


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

class EncoderToDecoderSkip(tf.keras.layers.Layer):
    """
    Block takes RGB and D features, applies spatial attention to both of them and then the result up.
    """
    def __init__(self, output_features_count, apply_attention_to_result=True):
        super(EncoderToDecoderSkip, self).__init__()
        self.output_features_count = output_features_count
        self.apply_attention_to_result=apply_attention_to_result
    def build(self,input_shape):
        #Note: grouping 1x1 conv may reduce computational complexity
        channel_count_rgb=int(self.output_features_count/2)
        channel_count_depth=self.output_features_count - channel_count_rgb
        self.conv1x1_first_rgb = tf.keras.layers.Conv2D(self.output_features_count, (1, 1),activation='relu')
        self.conv1x1_first_depth = tf.keras.layers.Conv2D(self.output_features_count, (1, 1),activation='relu')
        self.concat = tf.keras.layers.Concatenate()
        if self.apply_attention_to_result:
            self.final_multiply = tf.keras.layers.Multiply()
            self.conv1x1_attentional = tf.keras.layers.Conv2D(self.output_features_count, (1, 1),activation='sigmoid')

    def call(self,inputs):
        input_rgb,input_depth = inputs
        rgb = self.conv1x1_first_rgb(input_rgb)
        depth=self.conv1x1_first_depth(input_depth)
        x=self.concat([rgb,depth])
        if self.apply_attention_to_result:
            x_attention=self.conv1x1_attentional(x)
            x=self.final_multiply([x,x_attention])
        return x


class UniteAndDivideAttentionModule(tf.keras.layers.Layer):
    """
    Block first applies 1x1 conv to reduce channel size for RGb and D and then applies attention based on the fused rGB and D channels
    Block may be unnecessarily large.

    Used for fusing information between encoder branches (after every Resplacement Module)
    """
    def __init__(self, output_features_count):
        super(UniteAndDivideAttentionModule, self).__init__()
        self.output_features_count = output_features_count


    def build(self, input_shape):
        # Note: grouping 1x1 conv may reduce computational complexity
        channel_count=self.output_features_count
        self.conv1x1_rgb = tf.keras.layers.Conv2D(channel_count, (1, 1),activation='relu')
        self.conv1x1_depth = tf.keras.layers.Conv2D(channel_count, (1, 1),activation='relu')

        self.concat_united = tf.keras.layers.Concatenate()
        self.conv1x1_united_first = tf.keras.layers.Conv2D(int(channel_count/4), (1, 1),activation='relu')
        self.conv1x1_united_to_rgb = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')
        self.conv1x1_united_to_depth = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')
        self.multiply_to_rgb = tf.keras.layers.Multiply()
        self.multiply_to_depth = tf.keras.layers.Multiply()

    def call(self, inputs):
        rgb_in,depth_in=inputs
        rgb= self.conv1x1_rgb(rgb_in)
        depth=self.conv1x1_depth(depth_in)

        united = self.concat_united([rgb,depth])
        united = self.conv1x1_united_first(united)
        united_to_rgb = self.conv1x1_united_to_rgb(united)
        united_to_depth = self.conv1x1_united_to_depth(united)

        depth_out= self.multiply_to_depth([united_to_depth])
        rgb_out = self.multiply_to_rgb([united_to_rgb])

        return rgb_out,depth_out

class EncoderFusionBlock(tf.keras.layers.Layer):
    #takes features from RGB and D channel and calculates which features should be passed on to RGB and D channel respectively
    """
    Some more planing has to be done, this code is currently not complete.
    First attempt: use this for fusing at the end of encoder
    """
    def __init__(self, input_features_count):
        super(EncoderFusionBlock,self).__init__()
        self.input_featrues_count=input_features_count
    def build(self, input_shape):
        #define layers here
        self.initial_concat = tf.keras.layers.Concatenate()
        self.conv1x1_first = tf.keras.layers.Conv2D(self.input_featrues_count,(1,1),activation='relu')
        self.conv1x1_second = tf.keras.layers.Conv2D(int(self.input_featrues_count/16),(1,1),activation='relu')
        self.conv1x1_third = tf.keras.layers.Conv2D(self.input_featrues_count,(1,1),activation='sigmoid')
        self.final_multiply = tf.keras.layers.Multiply()
        self.final_add=tf.keras.layers.Add()

    def call(self,inputs):
        rgb_features,depth_features=inputs
        x = self.initial_concat([rgb_features,depth_features])
        x = self.conv1x1_first(x)
        x = self.conv1x1_second(x)
        x = self.conv1x1_third(x)
        x= self.final_multiply([x,depth_features])
        rgb_out= self.final_add([x,rgb_features])
        depth_out=x
        return rgb_out,depth_out


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

class ChannelSplitter(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ChannelSplitter, self).__init__(**kwargs)

    def build(self, input_shape):
        #self.reshape_r= tf.keras.layers.Reshape((256,256,1))
        #self.reshape_g= tf.keras.layers.Reshape((256,256,1))
        #self.reshape_b= tf.keras.layers.Reshape((256,256,1))
        self.reshape_d= tf.keras.layers.Reshape((256,256,1))

    def call(self, inputs):
       ## Split the input tensor into color channels.
       ##256,256,4
       #red_channel = inputs[..., 0]
       ##red_channel = tf.expand_dims(red_channel,-1)
       #red_channel = self.reshape_r(red_channel)
       #green_channel = inputs[..., 1]
       ##green_channel = tf.expand_dims(green_channel,-1)
       #green_channel = self.reshape_g(green_channel)
       #blue_channel =  inputs[..., 2]
       ##blue_channel =  tf.expand_dims(blue_channel,-1)
       #blue_channel =  self.reshape_b(blue_channel)
       depth_channel=  inputs[...,3]
       #depth_channel=  tf.expand_dims(depth_channel,-1)
       depth_channel=  self.reshape_d(depth_channel)
       #rgb=tf.concat([red_channel,green_channel,blue_channel], axis=-1)
       # Return a list of channel tensors.
       return inputs[..., 0:3], depth_channel


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




class ResplacmentUpsamplingBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), stride=(2, 2)):
        super(ResplacmentUpsamplingBlock, self).__init__()
        self.filters=filters
        self.kernel_size=kernel_size
        self.stride=stride
    def build(self, input_shape):
        self.tconv = tf.keras.layers.Conv2DTranspose(self.filters, self.kernel_size, strides=self.stride, padding='same')
        #self.tconv_1 = tf.keras.layers.Conv2DTranspose(self.filters, (self.kernel_size[0],1), strides=(self.stride[0],1), padding='same')
        #self.tconv_2 = tf.keras.layers.Conv2DTranspose(self.filters, (1,self.kernel_size[1]), strides=(1,self.stride[1]), padding='same')
        self.batchNorm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        #x=self.tconv_1(inputs)
        #x=self.tconv_2(x)

        x=self.tconv(inputs)

        x=self.batchNorm(x)
        x=self.relu(x)
        return x




def get_resnet_module(filters,input,perform_pooling=True):
    if perform_pooling:
        input = tf.keras.layers.MaxPooling2D()(input)
    rgb_branch = tf.keras.layers.Conv2D(filters, 3, strides=(1,1), padding='same')(input)
    tf.keras.layers.ReLU()(rgb_branch)
    rgb_branch = ResNetBlock(filters)(rgb_branch)
    rgb_branch = ResNetBlock(filters)(rgb_branch)
    rgb_branch = ResNetBlock(filters)(rgb_branch)
    return rgb_branch


def get_ResNet_model(OUTPUT_CLASSES=2):
    input_rgbd = tf.keras.layers.Input(shape=(256, 256, 4))
    input_rgb, input_d = ChannelSplitter()(input_rgbd)

    rgb_branch= tf.keras.layers.Conv2D(64, 3,strides=(2,2), padding='same')(input_rgb)
    rgb_branch = get_resnet_module(64, rgb_branch)  # 128x128
    rgb_branch = get_resnet_module(128, rgb_branch, perform_pooling=True)  # 64#64
    rgb_branch = get_resnet_module(256, rgb_branch, perform_pooling=False)  # 32x32
    rgb_branch = get_resnet_module(512, rgb_branch, perform_pooling=False)

    depth_branch= tf.keras.layers.Conv2D(64, 3,strides=(2,2), padding='same')(input_d)
    depth_branch = get_resnet_module(64, depth_branch)
    depth_branch = get_resnet_module(128, depth_branch, perform_pooling=True)
    depth_branch = get_resnet_module(256, depth_branch, perform_pooling=False)
    depth_branch = get_resnet_module(512, depth_branch, perform_pooling=False)

    # rgb_branch,depth_branch=blocks.EncoderFusionBlock(512)([rgb_branch,depth_branch])
    merged = tf.keras.layers.Concatenate()([rgb_branch, depth_branch])
    
    decoder = ResplacmentUpsamplingBlock(256)(merged)  # 64
    decoder = ResplacmentUpsamplingBlock(128)(decoder)
    decoder = ResplacmentUpsamplingBlock(64)(decoder)
    decoder = tf.keras.layers.Conv2D(filters=OUTPUT_CLASSES, kernel_size=(3, 3), padding='same')(decoder)
    decoder = tf.keras.layers.Softmax()(decoder)
    model = tf.keras.models.Model(inputs=input_rgbd, outputs=decoder)

    return model





