#B1 Final Project- Adaptive Deblurring Using Optimised Kernels
#Paras Shah

import numpy as np

def convolution_brute_force(input_image,kernel):
    """
    This function performs convolution of the input image with the kernel using the brute force method.

    
    Input Arguments:
    input_image: 2D numpy array
    kernel: 2D numpy array
    
    Returns:
    output_image: 2D numpy array

    Crop/overlap method (No padding used)
    """
    #Find input image size (n x n)
    input_image_size = input_image.shape

    #Find kernel size (m x m)
    kernel_size = kernel.shape

    #Initalise output image array of size; (n-m+1) x (n-m+1)
    output_image = np.zeros((input_image_size[0]-kernel_size[0]+1,input_image_size[1]-kernel_size[1]+1))
    
    #Iterate over the input image applying the kernel to every pixel
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):

            #Consider the region of the input image of size m x m (kernel size)
            region = input_image[i:i + kernel_size[0], j:j + kernel_size[1]]

            #Apply the kernel to the region via element-wise multiplication and sum the results
            output_image[i,j] = np.sum(region * kernel)

    return output_image