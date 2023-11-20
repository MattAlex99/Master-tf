
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np




def get_images_in_dictionary(origin_path,target_path,scaling_factor=1000):
    image_filenames = os.listdir(origin_path)
    image_paths = [os.path.join(origin_path, filename) for filename in image_filenames]

    for image_path in image_paths:
        exr_image=cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        png_image=(exr_image*scaling_factor).astype(np.uint16)

        image_name=image_path.split("\\")[-1][:-4]
        cv2.imwrite(target_path+"/"+image_name+".png",png_image)



if __name__ == "__main__":
    get_images_in_dictionary("C:\\Uni\\Master\\tensorflow_ws\\Datasets\\reference\\cleargrasp\\depth_exr",
                             "C:\\Uni\\Master\\tensorflow_ws\\Datasets\\reference\\cleargrasp\\depth",

                             )
