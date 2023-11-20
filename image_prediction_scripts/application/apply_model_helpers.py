import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

model = None
target_size=(256,256)

def predict_on_rgbd_image(rgb_image, depth_image, local_model):
    global model
    if model == None:
        model = local_model
    rgbd_image = tf.concat([rgb_image, depth_image], axis=-1)
    prediction_result = model.predict(rgbd_image)
    return prediction_result


def set_used_model(local_model):
    global model
    model = local_model


def load_model_from_path(path):
    global model
    model = tf.saved_model.load(path)
    return model


def load_depth_img_from_path(path):
    depth_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    global target_size
    depth_image = cv2.resize(depth_image,target_size)


    depth_image= tf.expand_dims(depth_image,-1)
    return depth_image


def load_rgb_image(path):
    rgb_image = cv2.imread(path)
    global target_size
    rgb_image =cv2.resize(rgb_image,target_size)
    return rgb_image


def stack_rgb_and_depth_image(rgb_img, depth_img):
    rgbd_tensor = tf.concat([rgb_img, depth_img], axis=-1)
    return rgbd_tensor


def get_rgbd_image_from_paths(rgb_path, depth_path):
    rgb_img = load_rgb_image(rgb_path)
    #rgb_img= cv2.cvtColor(rgb_img,cv2.COLOR_BGR2RGB)
    rgb_img=rgb_img.astype(np.float32)
    rgb_img=rgb_img/255
    depth_img = load_depth_img_from_path(depth_path)/1000
    rgbd_tensor = stack_rgb_and_depth_image(rgb_img, depth_img)
    return rgbd_tensor

def predict_on_image(rgbd_image,model_used=None):
    if model_used == None:
        global model
        model_used=model
    result= model_used([rgbd_image])

    return  result[0]

def display_prediction_result(rgbd_image,prediction):
    image = rgbd_image[..., 0:3]
    depth = tf.expand_dims(rgbd_image[..., 3], -1)

    prediction =create_mask(prediction)
    #prediction =prediction[...,2]
    #prediction=np.expand_dims(prediction,-1)
    display([image,depth,prediction])

def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask
def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'depth', 'Predicted Mask']

    for i in range(len(display_list)):
        print("display")
        plt.subplot(1, len(display_list), i + 1)
        print(np.shape(display_list[i]))
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

"""
Tests and usage
"""

if __name__ == "__main__":



    model = load_model_from_path("C:/Uni/Master/tensorflow_ws/models_trained/custom_architecture/grip_point_detection")
    set_used_model(model)
    #rgb_path=     "C:/Uni/Master/tensorflow_ws/Datasets/02_validated_datasets/ok/rgb/E6N8_0000350.png"
    #depth_path= "C:/Uni/Master/tensorflow_ws/Datasets/02_validated_datasets/ok/depth/E6N8_0000350.exr"

    for i in range (0,100):
        name_str= "0"*(7-len(str(i))) +str(i)
        print(name_str)
        rgb_path=     "C:/Uni/Master/tensorflow_ws/Datasets/02_validated_datasets/faulty/rgb/4JOA_"+name_str+".png"
        depth_path= "C:/Uni/Master/tensorflow_ws/Datasets/02_validated_datasets/faulty/depth/4JOA_"+name_str+".exr"

        try:
            rgbd_image=get_rgbd_image_from_paths(rgb_path,depth_path)

            result_image=predict_on_image(rgbd_image)

            display_prediction_result(rgbd_image,result_image)
        except:
            print("error")
            pass