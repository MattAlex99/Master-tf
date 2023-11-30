"""
This file takes a path to a lableed graspability dataset and applies random rotations, flips etc to the image to increase the amount of trainingsdata
"""
import matplotlib.pyplot as plt
import cv2
import helpers
import numpy as np
import random
from pathlib import Path

random.seed(42)


def save_image_pair(target_path, rgb, d, grasp, angle_image,image_count):
    print("Saving image with id {}".format(image_count))
    print(target_path)
    angle_image=angle_image/np.pi*255
    d=d*1000
    rgb=rgb*255
    grasp=grasp*255

    Path(target_path+"/rgb/").mkdir(parents=True, exist_ok=True)
    Path(target_path+"/depth/").mkdir(parents=True, exist_ok=True)
    Path(target_path+"/mask_grip/").mkdir(parents=True, exist_ok=True)
    Path(target_path+"/mask_angle/").mkdir(parents=True, exist_ok=True)

    rgb_path=target_path+"/rgb/"+str(image_count)+".png"
    d_path=target_path+"/depth/"+str(image_count)+".png"
    grasp_path=target_path+"/mask_grip/"+str(image_count)+".png"
    angle_path=target_path+"/mask_angle/"+str(image_count)+".png"

    cv2.imwrite(rgb_path,rgb.astype(np.uint8))
    cv2.imwrite(d_path,d.astype(np.uint16))
    cv2.imwrite(grasp_path,grasp.astype(np.uint8))
    cv2.imwrite(angle_path,angle_image.astype(np.uint8))


def generate_altered_images(rgb, d, grasp, horizontal, vertical, rotation_angle=25,flip_percent_h=0.5,flip_percent_w=0.0):
    angle_image = np.arctan2(vertical, horizontal)  # angles are in radians


    # apply_rotation to regular images
    rotation_angle = random.randint(-rotation_angle, rotation_angle)
    rgb = rotate_image(rgb, rotation_angle)
    d = rotate_image(d, rotation_angle)
    grasp= rotate_image(grasp,rotation_angle,"nearest")
    #apply_rotation_to_the_angle_image
    angle_image=rotate_image(angle_image,rotation_angle,"nearest")
    angle_image=np.mod(angle_image-np.radians(rotation_angle),np.pi)
    angle_image= angle_image*(grasp)

    #apply_flipps
    flip_thresh_horizontal=random.uniform(0,1)
    flip_thresh_vertical=random.uniform(0,1)
    if flip_thresh_horizontal<flip_percent_h:
        rgb = np.flip(rgb, 0)
        d= np.flip(d,0)
        grasp = np.flip(grasp, 0)
        angle_image = np.flip(angle_image, 0)
        angle_image = (angle_image- np.pi) *-1
        angle_image = angle_image * grasp

    if flip_thresh_vertical<flip_percent_w:
        rgb = np.flip(rgb, 1)
        d = np.flip(d, 1)
        grasp = np.flip(grasp, 1)
        angle_image = np.flip(angle_image, 1)
        angle_image = ((angle_image * -1)+np.pi)
        angle_image = angle_image * grasp
    return rgb,d,grasp,angle_image

def rotate_image(image, angle, inter="linear"):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    if inter == "linear":
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    else:
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
    return result


def generate_altered_dataset(origin_path, target_path, replication_rate=8):
    """
    Generates a new dataset (from origin path) and saves it in target_path, #replicationrate altered images are saved,
    image names are not kept
    :param origin_path:
    :param target_path:
    :param replication_rate:
    :return:
    """
    names_dataset = helpers.get_path_names_orientation_datset(origin_path, shuffle=True)
    names_dataset.shuffle(4)
    dataset = names_dataset.map(helpers.map_name_angle_dataset)
    image_count=0
    for i in names_dataset.as_numpy_iterator():
        print(i)
    for data in dataset.as_numpy_iterator():
        rgbd = data[0]
        rgb = rgbd[..., 0:3]
        depth = rgbd[..., -1]
        label = data[1]
        grasp = label[0]
        horizontal = label[1]
        vertical = label[2]
        angle_image = np.arctan2(vertical, horizontal)
        save_image_pair(target_path, rgb, depth, grasp, angle_image,image_count)
        image_count+=1
        for i in range(replication_rate):
            rgb_new,d_new,grasp_new,angle_image_new= generate_altered_images(rgb, depth, grasp, horizontal, vertical)
            save_image_pair(target_path,rgb_new,d_new,grasp_new,angle_image_new,image_count)
            image_count+=1


if __name__ == "__main__":
    origin_path = "C:\\Uni\\Master\\tensorflow_ws\\Datasets\\Datasets_grip_orientation\\002\\ok"
    target_path = "C:\\Uni\\Master\\tensorflow_ws\\Datasets\\Datasets_grip_orientation\\003"

    generate_altered_dataset(origin_path, target_path)
