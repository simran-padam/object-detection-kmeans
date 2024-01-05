import cv2
import os
import time
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

def closest(p, centers, norm = 2):
    """
    Compute closest centroid for a given point.
    Args:
        p (numpy.ndarray): input point
        centers (list): A list of center points of clusters
        norm (int): 2
    Returns:
        int: The index of closest centroid.
    """
    closest_c = min([(i, np.linalg.norm(p - c, norm))
                    for i, c in enumerate(centers)],
                    key=itemgetter(1))[0]
    return closest_c

def normalize(img):
    pixels = img.reshape((-1, 3))
    
    min_RGB = pixels.min(axis =0 )
    max_RGB = pixels.max(axis =0 )

    for pixel in pixels:
        pixel = 255 * (pixel - min_RGB) / (max_RGB-min_RGB)
    
    return img

def get_box(img):
    
    #convert from BGR to RGB
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #normalize the image to enhance the contrast of the image
    img = normalize(img) 

    #flatten the image 
    pixels = img.reshape((-1, 3))

    #clustering using k=10, iterations =20, 
    #accuracy of epsilon = 0.0003 and KMEANS_PP_CENTERS
    num_clusters = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.0003)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), num_clusters, None, criteria, 10,cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)
    
    #using red color to map the closest centers
    red = (210,0,0) 
    blue = (0,0,255)
    cluster_index = closest(red, centers, 2)

    #get a masked image 
    masked_image = np.copy(img)
    labels_reshape = labels.reshape(img.shape[0], img.shape[1])
    masked_image[labels_reshape == cluster_index] = [blue]

    #converting masked image to hsv image as it 
    #describes color using more familiar comparisons 
    #such as color, vibrancy and brightness
    hsv_img = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
    blue_bgr = np.uint8([[[255,0,0]]])
    hsv_blue = cv2.cvtColor(blue_bgr,cv2.COLOR_BGR2HSV)

    #creating a threshold using min and max of color and finding countours
    lower_blue = (120,255,250)
    upper_blue = (120,255,255)
    COLOR_MIN = np.array([lower_blue],np.uint8)
    COLOR_MAX = np.array([upper_blue],np.uint8)
    frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX) 
    imgray = frame_threshed
    ret,thresh = cv2.threshold(frame_threshed,127,255,0) 
    cv2.imshow("thresh",thresh)
    contours, hierarchy  = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    
    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]

    #creating a bounding rectangle and return dimensions
    x,y,w,h = cv2.boundingRect(cnt)

    pad_w = 3
    pad_h = 4
    pad_x = 3
    pad_y = 4

    xmin = x-pad_x
    ymin = y-pad_y
    xmax = x+w+pad_w
    ymax = y+h+pad_h

    return xmin, ymin, xmax, ymax

if __name__ == "__main__":

    start_time = time.time()
    counter = 0 

    dir_path = './images/'

    for i in range(1, 25):
        img_name = f'stop{i}.png'
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
      
        # Get the coordinators of the box
        xmin, ymin, xmax, ymax = get_box(img)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        output_path = f'./results/{img_name}'
        cv2.imwrite(output_path, img)
               
    end_time = time.time() #takes 28s to run
    # Make it < 30s
    print(f"Running time: {end_time - start_time} seconds")

