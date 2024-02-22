
"""
The purpose of this file is to load a trained model and to check its weights in different layers
to check the importance of local and global weights

"""
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

def plot_bias(bias):
    plt.hist(bias, density=False, bins=30)  # density=False would make counts
    plt.ylabel('value count')
    plt.xlabel('bias value')
    plt.show()

def plot_kernel(kernel):
    kernel = kernel.numpy().flatten()
    average=np.average(kernel)
    mean=np.median(kernel)
    print("average_value,",average)
    print("median_value,",mean)
    print("variance,",np.var(kernel))
    print("standart derivation,",np.std(kernel))
    plt.hist(kernel, density=False, bins=30)  # density=False would make counts
    plt.ylabel('value count')
    plt.xlabel('bias value')
    plt.show()

def process_encoder_fusion(bias=[],kernels=[]):
    kernel_values=np.array([])
    bias_values=np.array([])
    for b in bias:
        bias_values=np.append(bias_values,b.numpy())
    for kernel in kernels:
        kernel_values=np.append(kernel_values,kernel.numpy())
    print("all kernels",kernel_values)
    print("average_value,",np.average(kernel_values))
    print("mean_value,",np.median(kernel_values))
    plt.hist(kernel_values, density=False, bins=30)  # density=False would make counts
    plt.ylabel('value count')
    plt.xlabel('kernel value')
    plt.show()
    print("average_value,",np.average(bias_values))
    print("mean_value,",np.median(bias_values))
    plt.hist(bias_values, density=False, bins=30)  # density=False would make counts
    plt.ylabel('value count')
    plt.xlabel('bias value')
    plt.show()


def process_layer_stack(stack,bins=50):
    values = np.array([])
    for s in stack:
        values = np.append(values, s.numpy())
    print("all kernels", values)
    print("average_value,", np.average(values))
    print("median_value,", np.median(values))
    print("variance,",np.var(values))
    print("standart derivation,",np.std(values))
    plt.hist(values, density=False, bins=bins)  # density=False would make counts
    plt.ylabel('value count')
    plt.xlabel('value')
    plt.show()

def separate_fully_and_semi_local_dense_layers(dense_layers,fully_local_postfixes=["/"],semi_local_postfixes=["_1/"]):
    #0,1 is local, 2,3 are seim_local
    result_fully=[]
    result_semi=[]
    for v in dense_layers:
        for postfix in fully_local_postfixes:
            if "squeeze_and_excitation_block"+postfix in v.name:
                result_fully.append(v)
        for postfix in semi_local_postfixes:
            if "squeeze_and_excitation_block"+postfix in v.name:
                result_semi.append(v)
    print("semi_locals:",len(result_semi))
    return result_fully,result_semi


def get_encoder_fusion_layers(model):
    kernels=[]
    bias=[]
    for v in model.trainable_variables:
        if ("encoder_fusion_block" in v.name) and ("kernel" in v.name):
            print("found kernel:", v.name)
            kernels.append(v)
        if ("encoder_fusion_block" in v.name) and ("bias" in v.name):
            print("found bias:", v.name)
            bias.append(v)
    print(len(kernels))
    print(len(bias))
    return kernels,bias
def get_resplacement_layers(model):
    kernels=[]
    bias=[]
    for v in model.trainable_variables:
        if ("resplacement_block" in v.name) and ("squeeze_and_excitation_block" in v.name) and ("dense" in v.name) and (
                "kernel" in v.name):
            print("found kernel:", v.name)
            kernels.append(v)
        if ("resplacement_block" in v.name) and ("squeeze_and_excitation_block" in v.name) and ("dense" in v.name) and (
                "bias" in v.name):
            print("second")
            bias.append(v)

    print(len(kernels))
    fully_local_kernels,semi_local_kernels= separate_fully_and_semi_local_dense_layers(kernels)
    fully_local_bias,semi_local_bias= separate_fully_and_semi_local_dense_layers(bias)
    print(len(fully_local_kernels))
    print("here")

    return semi_local_kernels, semi_local_bias, fully_local_kernels,fully_local_bias

if __name__=="__main__":
    """
    If you cant install graviz on your device, you can still run this script, if you set plot model to false
    """

    plot_model=True
    model_path = "../../../models_trained/architecture_graspabiltiy/test_more_local_non_normalized_max_distance800_no_add_attention_to_rgb"
    images_folder="../../images/"
    model = load_model(model_path)
    print(model.summary())

    kernels=[]
    bias= []
    if plot_model:
        tf.keras.utils.plot_model(
            model,
            to_file=images_folder+'./model.png',
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=96,
            layer_range=None,
            show_layer_activations=True,
            show_trainable=False
        )

    for v in model.trainable_variables:
        print(v.name)

    encoder_fusion_kernels=[]
    encoder_fusion_bias=[]


    fusion_kernel,fusion_bias=get_encoder_fusion_layers(model)
    print("encoder_fusion_kernels")
    process_layer_stack(fusion_kernel,bins=50)
    print("encoder_fusoin bias")
    process_layer_stack(fusion_bias,bins=50)


    sorted_layers=get_resplacement_layers(model)
    semi_local_kernels, semi_local_bias, fully_local_kernels, fully_local_bias = sorted_layers

    print("semi_loca_bias:")
    process_layer_stack(semi_local_bias)
    print("fully_local_bias")
    process_layer_stack(fully_local_bias)

    print("semi_loca_kernel:")
    process_layer_stack(semi_local_kernels)
    print("fully_local_kernel")
    process_layer_stack(fully_local_kernels)

    #for kernel in semi_local_kernels:
    #    print("plotting kernel")
    #    print(kernel.numpy())
    #    plot_kernel(kernel)





