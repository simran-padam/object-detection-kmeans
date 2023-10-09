
## GR5293 Applied Machine Learning for Computer Vision (HW 1)

#### Object Detection using K-means
*The purpose of the code is to detect the "STOP" sign in 24 distinct road images using K-means.* 


#### Pre-requisites

- *Python*
- *Install libraries: time, operator, os, numpy, matplotlib, cv2*
- *Download the images folder and run the code on a local environment*


### Introduction: K-means algorithm 

*Using OpenCV, K-means clustering can be implemented to segment the image into "k" clusters. cv2.kmeans() has parameters such as accuracy, number of iterations, number of clusters, etc. Using such criteria, the code attempts to segment the image.* 

*The following result depicts the segmented image after running kmeans.*

<img src="https://raw.githubusercontent.com/simran-padam/object-detection-kmeans/main/kmeans-image.png"  width="400" height="300">

*source: <https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html>*


### K-means implementation

*The following summary details the parameters created and used to implement the K-means clustering technique.*

#### Description of parameters

| Name                       | Parameter      | Description                                                                 |
| -------------------------- | ------------------ | --------------------------------------------------------------------------- |
| cv2.kmeans               | TERM_CRITERIA_EPS | it stops the algorithm iteration if the specified accuracy, epsilon, is reached. TERM_CRITERIA_EPS = 0.0003 is used. It specifies the accuracy you require in your subpixel values, for example, a value of 0.0001 means you are asking for subpixel values down to an accuracy of 1/10000th of a pixelname                                                          |
| cv2.kmeans                  | TERM_CRITERIA_MAX_ITER | limits the number of iterations in the k-means algorithm. Iteration stops after this many iterations even if the convergence criterion is not satisfied.  TERM_CRITERIA_MAX_ITER = 20 is used to keep processing time below 30s. More iterations means more time to run the model              |
| cv2.kmeans              | KMEANS_PP_CENTERS                 | the method first iterates the whole image to determine the probable centers and then starts to converge                                        |
| closest()            | p            | user-defined function - parameter "p" is the input point for which we want to find the closest point from centers. Red pixel points are used as "p"|
| closest()                   | norm    | l2 norm - euclidean distance                                                               |
| closest()         | centers       | centers of each cluster                                                        |
| normalize()             | img              | user-defined function to normalise the image to improve the contrast the image and reduce the scale differences between features which leads to faster convergence                                     |


#### Caution:

Please keep in mind while implementing kmeans clustering: 

* Using a higher number of iterations in kmeans will increase the processing time of the model and impact model performnace
* Choosing lower value of k is also prone to affect the performance. For ex: if there are more colors in an image, a lower value of k might not result in good segmentation. K-means is ill-suited for tasks where the number of objects is not known beforehand and sometimes choosing the right k can be challenging.
* kmeans can falter significantly if the data points are high as it maps each point with the other data point to calculate distance, the processing time will be high and the code might fail to run.
* Enabling cv.KMEANS_RANDOM_CENTERS flag results in new output each run, reduces the chances of reproducibility, and this can potentially impact the model performance.

Citation:
* *Data Source: https://www.kaggle.com/datasets/andrewmvd/road-sign-detection*

* https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
