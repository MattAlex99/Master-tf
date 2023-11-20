import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter

import itertools

def load_depth_map_from_path(path):
    depth_image= cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)

    if len(np.shape(depth_image))>=3:
        if len(depth_image[0][0]) ==3:
            depth_image= cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    depth_image=cv2.resize(depth_image,(256,144))
    return depth_image


def display_depth_image(img):
    imgplot = plt.imshow(img)
    plt.show()

def get_normals_from_depth_GPT(depth_image):

    # Calculate gradients in the x and y directions using Sobel filters
    dx = cv2.Sobel(depth_image, cv2.CV_32F, 1, 0, ksize=5)
    dy = cv2.Sobel(depth_image, cv2.CV_32F, 0, 1, ksize=5)

    # Calculate the surface normals from the gradients
    normals = np.dstack((-dx, -dy, np.ones_like(depth_image)))  # Note: Negate dx and dy for correct orientation
    # Normalize the normals to get unit vectors
    normals_length = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
    unit_normals = normals / normals_length
    return unit_normals


def get_normals_from_depth_STACKOVERLOW(depth_image):
    zy, zx = np.gradient(depth_image)
    # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
    # to reduce noise
    #zx = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=5)
    #zy = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=5)

    normal = np.dstack((-zx, -zy, np.ones_like(depth_image)))

    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    return normal
def get_suction_gripable_points_from_normals(normals_image,surface_tollerance=0.1,kernel_size=10):
    #surface_tollerance_describes how different the surface normaly may be before a position is deemed to be not usefull
    def dot_product_counter_kernel(a,threshhold=1-surface_tollerance):
        result=0
        #get the center pixel
        normal_1_index=kernel_size ** 2 // 2 + 1
        normal_1 = [a[normal_1_index],a[normal_1_index+int(kernel_size**2)],a[normal_1_index+int(kernel_size**2)]]

        for i in range(int(kernel_size**2)):
            #get the second pixel for compatison
            normal_2=[a[i],a[i+int(kernel_size**2)],a[i+int(kernel_size**2)]]
            if np.dot(normal_1,normal_2)<threshhold:
                result+=1
        return 1 if result>0.9*kernel_size**2 else 0

    #footprint = np.array([[True, True, True],
    #                      [True, True, True],
    #                      [True, True, True]])
    suction_points = generic_filter(normals_image, dot_product_counter_kernel,size=(kernel_size,kernel_size,3)) #footprint can be used to decide which kernel points are going to be used, may be good to reduce for performance reasons
    print("suction:",np.shape(suction_points))
    return suction_points[...,0]

def get_indexes_around_point(position=(50,50),initial_step_range=2,image_size=(100,100),mode="random"):
    """
    gets the indexes around a positionin a star patern (8 cardinal directions) with increasing distance
    :param position: point around which indexes are retrieved
    :param initial_step_range: rate how far the field of view will be
    :return: a list of indexes
    """
    if mode=="random":
        indexes_helper=range(-initial_step_range,initial_step_range,2)
        #indexes_helper=[initial_step_range,initial_step_range*2,initial_step_range*4,-initial_step_range,-initial_step_range*2,-initial_step_range*4]
        indexes=[]
        for element in itertools.product(indexes_helper,indexes_helper):
            x_cord=element[0]+position[0]
            y_coord= element[1]+position[1]
            if x_cord<0 or x_cord>=image_size[0]:
                return None
            if y_coord<0 or y_coord>=image_size[1]:
                return None
            indexes.append( [x_cord, y_coord])
        return indexes
    if mode=="lines":
        indexes_helper=range(initial_step_range)
        line1=[(position[0]+ i, position[1]) for i in indexes_helper ]
        line2=[(position[0]+ i, position[1]+i) for i in indexes_helper ]
        line3=[(position[0],    position[1]+i) for i in indexes_helper ]
        line4=[(position[0]-i,  position[1]+i) for i in indexes_helper ]
        line5=[(position[0]-i,  position[1]) for i in indexes_helper ]
        line6=[(position[0]-i,  position[1]-i) for i in indexes_helper ]
        line7=[(position[0],    position[1]-i) for i in indexes_helper ]
        line8=[(position[0]+i,  position[1]-i) for i in indexes_helper ]
        result=[]
        result.append(line1)
        result.append(line2)
        result.append(line3)
        result.append(line4)
        result.append(line5)
        result.append(line6)
        result.append(line7)
        result.append(line8)
        for line in result:
            for point in line:
                x_cord,y_coord=point
                if x_cord < 0 or x_cord >= image_size[0]:
                    return None
                if y_coord < 0 or y_coord >= image_size[1]:
                    return None
        return result
    return -1

def get_suctionability_at_position(normals,centre_position=(50,50),initial_step_range=2,tolerance=0.1,mask=None):
    #initial_step_range determines how far pixels which are observed are apart
    #with range 2 the following pixels will be checked:
    # X010001000000001
    #pixels in 8 cardnial directions are checked
    #get indexes of points that need to be checked:
    max_size=np.shape(normals)
    mode="random"
    if mode =="lines":
        all_indexes = get_indexes_around_point(position=centre_position, initial_step_range=initial_step_range,
                                               image_size=max_size,mode=mode)
        if all_indexes==None:
            return 0
        score=0
        point_of_comparisson = centre_position
        for index_line in all_indexes:
            line_score=0
            set_to_0_helper=1
            for index in index_line:
                normal1 = normals[point_of_comparisson[0], point_of_comparisson[1]]
                normal2 = normals[index[0], index[1]]
                dot_product = np.dot(normal1, normal2)
                #print("dot:", dot_product)
                line_score+=dot_product**2
                if dot_product< 0.5:
                    if set_to_0_helper ==0:
                        set_to_0_helper=-1
                    set_to_0_helper=0
            point_of_comparisson=index
            line_score = line_score * set_to_0_helper
            score+= line_score
            #print("line",line_score)

        score=score/ ( len(all_indexes[0])*len(all_indexes[0]) )
        #print(score ,"\n\n")
        return score
    if mode=="random":
        all_indexes=get_indexes_around_point(position=centre_position,initial_step_range=initial_step_range,image_size=max_size)
        if all_indexes ==None:
            #return 0 if the index is out of range
            return 0
        matches=0
        score=0
        for index in all_indexes:
            normal1=normals[centre_position[0],centre_position[1]]
            normal2=normals[index[0],index[1]]

            dot_product= np.dot(normal1,normal2)**2 *mask[index[0]][index[1]]
            score+= dot_product
            if dot_product<0.9:
                score-=1
            if dot_product > 1-tolerance:
                matches+=1

        score=score/len(all_indexes)
        return score

def get_suction_points_sparse(normals,mask):
    #TODO is not sparse yet, is performed for every pixel
    result=np.zeros_like(normals)
    width=len(normals[0])
    height=len(normals)
    for index in itertools.product(range(0,height-1),range(0,width-1)):
        value=get_suctionability_at_position(normals,centre_position=index,initial_step_range=5,mask=mask)
        result[index[0],index[1]]=value
    return result

def get_bounding_box_coordinates_from_mask(mask):
    """
    takes a binary image mask and returns the coordinates of all bounding boxes for all objects
    :param mask:
    :return: a list of coodinates
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0
    result=[]
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        result.append([(x,y),(x+w,y+h)])
    return result
def get_region_of_interest_from_mask(mask,image,context_size=10):
    """

    :param mask: the image mask
    :param context_size:
    :return:
    """
    bboxes=get_bounding_box_coordinates_from_mask(mask)
    result=[]
    for bbox in bboxes:
        x_top_left= bbox[0][0]-context_size
        y_top_left= bbox[0][1]-context_size
        x_bottom_right=bbox[1][0]+context_size
        y_bottom_right=bbox[1][1]+context_size

        roi_img = image[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
        roi_mask= mask[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
        roi_mask=np.expand_dims(roi_mask,-1)
        result.append([roi_img,roi_mask])
    return result
#image= load_depth_map_from_path("C:\\Uni\\Master\\tensorflow_ws\\Master-tf\\images\\normals_test.png")
#display_depth_image(image)
#
#calculated_normals= get_normals_from_depth_GPT(image)
#display_depth_image(calculated_normals)
#
#calculated_normals2= get_normals_from_depth_STACKOVERLOW(image)
#display_depth_image(calculated_normals2)




image= load_depth_map_from_path("C:\\Uni\\Master\\tensorflow_ws\\Datasets\\raw\\cleargrasp\\cleargrasp-dataset-test-val\\cleargrasp-dataset-test-val\\real-test\\d415\\000000045-output-depth.exr")
image=image*255 #depending on how they are extracted, image must be in range 0-1 or 0-255
display_depth_image(image)

mask= load_depth_map_from_path("C:\\Uni\\Master\\tensorflow_ws\\Datasets\\raw\\cleargrasp\\cleargrasp-dataset-test-val\\cleargrasp-dataset-test-val\\real-test\\d415\\000000045-mask.png")
display_depth_image(mask)
get_bounding_box_coordinates_from_mask(mask)

rois= get_region_of_interest_from_mask(mask=mask,image=image)
for roi in rois:
    roi_img,roi_mask=roi
    roi_mask=roi_mask

    #roi_img=cv2.normalize(np.array(roi_img), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #roi_img=roi_img*np.squeeze(roi_mask,-1)/255

    #show rois
    display_depth_image(roi_img)

    #calculate how gripable they are
    calculated_normals = get_normals_from_depth_STACKOVERLOW(roi_img)
    calculated_normals= calculated_normals* roi_mask/255
    display_depth_image(calculated_normals)


    sparse_suction_points = get_suction_points_sparse(calculated_normals,roi_mask/255) /255 * roi_mask
    #sparse_suction_points=cv2.normalize(sparse_suction_points, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    display_depth_image(sparse_suction_points)

#show for toal image
calculated_normals= get_normals_from_depth_STACKOVERLOW(image)
display_depth_image(calculated_normals)


sparse_suction_points=get_suction_points_sparse(calculated_normals,np.ones_like(calculated_normals)[...,0])
#sparse_suction_points=get_suction_points_sparse(calculated_normals,mask/255)
display_depth_image(sparse_suction_points)


#suction_map=get_suction_gripable_points_from_normals(calculated_normals)
#display_depth_image(suction_map)
#
#image_normals= load_depth_map_from_path("C:\\Uni\\Master\\tensorflow_ws\\Datasets\\raw\\cleargrasp\\cleargrasp-dataset-test-val\\cleargrasp-dataset-test-val\\real-test\\d415\\000000000-normals.exr")
#display_depth_image(image_normals)

