## Do elastic transformation
##
## source: https://github.com/vsvinayak/mnist-helper/blob/master/mnist_helpers.py
##

import cv2
import numpy
import math
import random

from numpy.random import random_integers
from scipy.signal import convolve2d



def create_2d_gaussian(dim, sigma):
    """
    This function creates a 2d gaussian kernel with the standard deviation
    denoted by sigma
    
    :param dim: integer denoting a side (1-d) of gaussian kernel
    :type dim: int
    :param sigma: the standard deviation of the gaussian kernel
    :type sigma: float
    
    :returns: a numpy 2d array
    """

    # check if the dimension is odd
    if dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # initialize the kernel
    kernel = numpy.zeros((dim, dim), dtype=numpy.float16)

    # calculate the center point
    center = dim/2

    # calculate the variance
    variance = sigma ** 2
    
    # calculate the normalization coefficeint
    coeff = 1. / (2 * variance)

    # create the kernel
    for x in range(0, dim):
        for y in range(0, dim):
            x_val = abs(x - center)
            y_val = abs(y - center)
            numerator = x_val**2 + y_val**2
            denom = 2*variance
            
            kernel[x,y] = coeff * numpy.exp(-1. * numerator/denom)
    
    # normalise it
    return kernel/sum(sum(kernel))




def elastic_transform(image, kernel_dim=13, sigma=6, alpha=36, negated=False):
    """
    This method performs elastic transformations on an image by convolving 
    with a gaussian kernel.
    NOTE: Image dimensions should be a sqaure image
    
    :param image: the input image
    :type image: a numpy nd array
    :param kernel_dim: dimension(1-D) of the gaussian kernel
    :type kernel_dim: int
    :param sigma: standard deviation of the kernel
    :type sigma: float
    :param alpha: a multiplicative factor for image after convolution
    :type alpha: float
    :param negated: a flag indicating whether the image is negated or not
    :type negated: boolean
    :returns: a nd array transformed image
    """
    
    # convert the image to single channel if it is multi channel one
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check if the image is a negated one
    if not negated:
        image = 255-image

    # check if the image is a square one
    if image.shape[0] != image.shape[1]:
        raise ValueError("Image should be of sqaure form")

    # check if kernel dimesnion is odd
    if kernel_dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # create an empty image
    result = numpy.zeros(image.shape)

    # create random displacement fields
    displacement_field_x = numpy.array([[random_integers(-1, 1) for x in xrange(image.shape[0])] \
                            for y in xrange(image.shape[1])]) * alpha
    displacement_field_y = numpy.array([[random_integers(-1, 1) for x in xrange(image.shape[0])] \
                            for y in xrange(image.shape[1])]) * alpha

    # create the gaussian kernel
    kernel = create_2d_gaussian(kernel_dim, sigma)

    # convolve the fields with the gaussian kernel
    displacement_field_x = convolve2d(displacement_field_x, kernel)
    displacement_field_y = convolve2d(displacement_field_y, kernel)

    # make the distortrd image by averaging each pixel value to the neighbouring
    # four pixels based on displacement fields
    
    for row in xrange(image.shape[1]):
        for col in xrange(image.shape[0]):
            low_ii = row + int(math.floor(displacement_field_x[row, col]))
            high_ii = row + int(math.ceil(displacement_field_x[row, col]))

            low_jj = col + int(math.floor(displacement_field_y[row, col]))
            high_jj = col + int(math.ceil(displacement_field_y[row, col]))

            if low_ii < 0 or low_jj < 0 or high_ii >= image.shape[1] -1 \
               or high_jj >= image.shape[0] - 1:
                continue

            res = image[low_ii, low_jj]/4 + image[low_ii, high_jj]/4 + \
                    image[high_ii, low_jj]/4 + image[high_ii, high_jj]/4

            result[row, col] = res
    
    # if the input image was not negated, make the output image also a non 
    # negated one
    if not negated:
        result = 255-result

    return result





##
## cropping and resizing image to (size_x, size_y):
##
## img2 - should be matrix
## img_x - new x shape dimension
## img_y - new y shape dimension

def crop_and_resize_img (img2,img_x,img_y):

	img_crop = numpy.zeros([img_x,img_y])

	cur_x = img2.shape[0]
	cur_y = img2.shape[1]


	# cropping image borders
	if (cur_y>img_y):
	    crop_y = (cur_y-img_y)/2.0
	    crop_y = int(math.ceil(crop_y))
	    img2 = img2[:,crop_y:(crop_y+28)]
	    
	# cropping image borders
	if (cur_x>img_x):
	    crop_x = (cur_x-img_x)/2.0
	    crop_x = int(math.ceil(crop_x))
	    img2 = img2[crop_x:(crop_x+28),:]


	# resizing smaller image to larger scale
	cur_x = img2.shape[0]
	cur_y = img2.shape[1]

	st_x = int(math.ceil((img_x - cur_x)/2.0-0.5))
	st_y = int(math.ceil((img_y - cur_y)/2.0-0.5))

	img_crop[st_x:(st_x+cur_x),st_y:(st_y+cur_y)] = img2
	return (img_crop)





##
##  scalling image by both coordinates by phi values:
##
## img_nes should be a matrix
##

def img_xy_scale (img_nes):
	
	phi_x = random.uniform(1-0.2,1+0.2)  # random scalling factor
	phi_y = random.uniform(1-0.2,1+0.2)  # random scalling factor
	res = cv2.resize(img_nes, None, fx=phi_x, fy=phi_y, interpolation = cv2.INTER_LINEAR)
	return res





##
##  rotating image by both coordinates by phi values:
##
## img_nes should be a matrix
##

def img_rotation (img_nes):
	
	beta = random.uniform(-15,15)  # random rotation coeffieint
	M = cv2.getRotationMatrix2D((img_nes.shape[0]/2,img_nes.shape[1]/2),beta,1)
	img_nes = cv2.warpAffine(img_nes,M,(img_nes.shape[0],img_nes.shape[1]))
	return img_nes