import numpy as np
from PIL import Image
from pylab import *

"""
Applying the first phase of the  Birch algorithm in an image segmentation 
in this method we used 2 structures : 
class_tb : an array that contains the classes; the index of this array is the class number and the content is the class representant 
val_tb : values array; the index of this array is the the matrix value and the content is the class that the value belong to 
-the birch method loops through the matrix and check if we are on the first pixel then create a class for it (append it to the class_tb array 
and its value to the val_tb array), if not then we check if the distance of the pixel is less then the threshold then we 
assign it to that class 
-then we calculate the centroid of each class and assign it to that specific index in class_tb
we assign new val_tb values(the index is the pixel and the value is how many times the pixel is repeated) from the classes centroids 
-then we assign the new pixel values from the histogram (val_tb)
"""


def birch(img, FT):
    # converting the img into an np array
    npImg = np.array(img, dtype='int16')
    # initialising arrays
    class_tb = []
    val_tb = np.ones((256, 1)) * -1
    # looping through np image pixels
    for i in range(npImg.shape[0]):
        for j in range(npImg.shape[1]):
            # checking if the class_tb is empty
            if class_tb is None:
                # appending the first pixel value to the classes array class_tb
                class_tb.append(npImg[i, j])
                # setting the value which's index is equal to the first pixel 0,  class 0 (first class)
                val_tb[npImg[i, j]] = 0
            else:
                # if the class_tb is not empty
                min_distance = FT
                class_id = -1
                # checking if the value is not verified (-1 means that the pixel is not classified)
                if val_tb[npImg[i, j]] == -1:
                    # looping through the class_tb array
                    for k in range(len(class_tb)):
                        # calculating the distance of the pixel to the class representant
                        distance = abs(npImg[i, j] - class_tb[k])
                        # we keep the smallest distance along with its associated class
                        if distance <= FT and distance <= min_distance:
                                min_distance = distance
                                class_id = k
                    # we assign the pixel to an existing class
                    if class_id != -1:
                        val_tb[npImg[i, j]] = class_id
                    else:
                        # we create a new class to the pixel
                        class_tb.append(npImg[i, j])
                        val_tb[npImg[i, j]] = len(class_tb) - 1

    # calculating the centroids of the classes
    class_tb = calculate_centroids(class_tb, val_tb)

    # affecting new values to the val_tb array
    val_tb = affect_new_vals(val_tb, class_tb)

    # calling the segmentation method
    npImg = segmentation(npImg, val_tb)

    return npImg


def birch_pondere(img, FT):
    # converting image to matrix
    npImg = np.array(img, dtype='int16')
    # initialising the arrays
    class_tb = []
    # val_tb is a two dimension array, the second row contains the occurance of the pixels
    val_tb = np.ones((256, 2)) * -1
    # initializing all occurances with 0
    val_tb[:, 1] = 0
    for i in range(npImg.shape[0]):
        for j in range(npImg.shape[1]):
            # affecting first image element to the class if the class is empty
            if class_tb is None:
                # appending the first pixel value to the classes array class_tb
                class_tb.append(npImg[i, j])
                # setting the value which's index is equal to the first pixel 0,  class 0 (first class)
                val_tb[npImg[i, j], 0] = 0
            else:
                min_distance = FT
                class_id = -1
                # if the value is not verified (-1 means that the pixel is not classified)
                if val_tb[npImg[i, j], 0] == -1:
                    for k in range(len(class_tb)):
                        distance = abs(npImg[i, j] - class_tb[k])
                        # if the distance is smaller than the threshold
                        if distance <= FT and distance <= min_distance:
                            min_distance = distance
                            class_id = k

                    # we assign the pixel to an existing class
                    if class_id != -1:
                        val_tb[npImg[i, j], 0] = class_id
                        val_tb[npImg[i, j], 1] += 1
                    else:
                        # we create a new class to the pixel
                        class_tb.append(npImg[i, j])
                        val_tb[npImg[i, j], 0] = len(class_tb) - 1
                        val_tb[npImg[i, j], 1] += 1
                else:
                    val_tb[npImg[i, j], 1] += 1

    # calculating the centroids of the classes
    class_tb = calculate_centroids_pondere(class_tb, val_tb)

    # affecting new values to the val_tb array
    val_tb = affect_new_vals_pondere(val_tb, class_tb)

    # calling the segmentation method
    npImg = segmentation_pondere(npImg, val_tb)

    return npImg


# function that calculates the centroids
def calculate_centroids_pondere(val_class, vals):
    # looping through the class array
    for i in range(len(val_class)):
        sum = 0
        items_number = 0
        # looping through the vals array
        for j in range(len(vals)):
            # storing the pixels that belong to the same class in the sum variable and calculating their number
            if vals[j, 0] == i:
                sum += j * vals[j, 1]
                items_number += vals[j, 1]
                # calculating centroids of the class and setting it to class array index
        val_class[i] = sum // items_number

    return val_class


# function that calculates the centroids
def calculate_centroids(val_class, vals):
    # looping through the class array
    for i in range(len(val_class)):
        sum = 0
        items_number = 0
        # looping through the vals array
        for j in range(len(vals)):
            # storing the pixels that belong to the same class in the sum variable and calculating their number
            if vals[j] == i:
                sum += j
                items_number += 1
        # calculating centroids of the class and setting it to class array index
        val_class[i] = sum // items_number

    return val_class


# affecting new values to the values table
def affect_new_vals_pondere(val, val_class):
    # looping through the
    for i in range(len(val)):
        # assigning new values to the val array from the class centroids
        val[i, 0] = val_class[int(val[i, 0])]
    return val


# affecting new values to the values table
def affect_new_vals(val, val_class):
    # looping through the
    for i in range(len(val)):
        # assigning new values to the val array from the class centroids
        val[i] = val_class[int(val[i])]
    return val


# segmentation function (assigning new pixel values)
def segmentation_pondere(img, val):
    # looping through the image pixels
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # assigning img new pixel values from val array
                img[i, j] = val[img[i, j], 0]
    return img


# segmentation function (assigning new pixel values)
def segmentation(img, val):
    # looping through the image pixels
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # assigning img new pixel values from val array
                img[i, j] = val[img[i, j]]
    return img


# main program
imgNDG = Image.open('image.jpg').convert('L')
Image.fromarray(birch(imgNDG, 20).astype('uint8')).save('result.jpg')