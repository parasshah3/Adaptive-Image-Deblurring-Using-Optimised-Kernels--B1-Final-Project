# B1 Final Project - Adaptive Deblurring Using Optimised Kernels
# Paras Shah

import numpy as np

def convolution_brute_force(input_image, kernel):
    """
    Perform convolution of the input image with the kernel using the brute force method.

    Input Arguments:
        input_image: 2D numpy array
        kernel: 2D numpy array

    Returns:
        output_image: 2D numpy array

    Crop/overlap method (No padding used).
    """
    # Check if the kernel dimensions are odd-sized
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Kernel must have odd dimensions. Current kernel size: "
                         f"{kernel.shape}")

    # Find dimensions of the input image and kernel
    input_image_size = input_image.shape
    kernel_size = kernel.shape

    # Flip the kernel in both directions
    flipped_kernel = np.flip(np.flip(kernel, axis=0), axis=1)

    # Initialise output image array of size (n-m+1) x (n-m+1)
    output_image = np.zeros((input_image_size[0] - kernel_size[0] + 1, input_image_size[1] - kernel_size[1] + 1))
    
    # Iterate over the input image applying the kernel to every region
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            # Extract the region of the input image of size equal to the kernel
            region = input_image[i:i + kernel_size[0], j:j + kernel_size[1]]
            
            # Apply the flipped kernel to the region via element-wise multiplication and sum the results
            output_image[i, j] = np.sum(region * flipped_kernel)

    return output_image


import numpy as np
from scipy.fft import fft2, ifft2

import numpy as np
from scipy.fft import fft2, ifft2

def convolution_AI(input_image, kernel):
    """
    Perform 2D convolution of an input image with a kernel using the FFT method.
    Handles zero-padding and alignment to ensure results match spatial convolution.

    Input Arguments:
        input_image: 2D numpy array
        kernel: 2D numpy array (must be odd-sized)

    Returns:
        output_image: 2D numpy array (filtered/convolved image array)
    """
    # Get dimensions of the input image and kernel
    input_height, input_width = input_image.shape
    kernel_height, kernel_width = kernel.shape

    # Check if the kernel dimensions are odd-sized
    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise ValueError("Kernel must have odd dimensions. Current kernel size: "
                         f"{kernel.shape}")

    # Determine the size for zero-padding (input + kernel - 1)
    pad_height = input_height + kernel_height - 1
    pad_width = input_width + kernel_width - 1

    # Zero-pad the input image and kernel to the same size
    padded_image = np.pad(input_image, ((0, pad_height - input_height), (0, pad_width - input_width)))
    padded_kernel = np.pad(kernel, ((0, pad_height - kernel_height), (0, pad_width - kernel_width)))

    # Perform FFT on the padded image and kernel
    fft_image = fft2(padded_image)
    fft_kernel = fft2(padded_kernel)

    # Multiply in the frequency domain
    fft_result = fft_image * fft_kernel

    # Inverse FFT to get back to the spatial domain
    convolved_image = np.real(ifft2(fft_result))

    # Compute valid region dimensions
    valid_height = input_height - kernel_height + 1
    valid_width = input_width - kernel_width + 1
    start_row = kernel_height - 1
    start_col = kernel_width - 1

    # Extract the valid region from the convolved image
    output_image = convolved_image[start_row:start_row + valid_height, start_col:start_col + valid_width]

    return output_image