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

from PIL import Image
import numpy as np

def load_image(filepath, grayscale=True):
    """
    Load an image and convert it to a numpy array.
    Args:
        filepath: Path to the image file.
        grayscale: Whether to convert the image to grayscale.

    Returns:
        image_array: 2D numpy array of the image.
    """
    img = Image.open(filepath)
    if grayscale:
        img = img.convert("L")  # Convert to grayscale
    image_array = np.array(img)
    return image_array

import numpy as np
from scipy.ndimage import sobel

def get_global_properties(image):
    """
    Computes the global properties of the image: resolution, global variance, and global gradient magnitude.

    Args:
        image (numpy.ndarray): The input image as a 2D or 3D array (grayscale or RGB).

    Returns:
        tuple: A tuple containing:
            - resolution (tuple): Tuple (height, width) of the image.
            - global_variance (float): Global variance of the image intensity.
            - global_gradient_magnitude (float): Average gradient magnitude of the image.
    """
    # Step 1: Compute resolution
    height, width = image.shape[:2]

    # Step 2: Compute global variance
    if len(image.shape) == 3:  # RGB Image
        image_gray = np.mean(image, axis=2)  # Convert to grayscale by averaging channels
    else:
        image_gray = image  # Grayscale Image

    global_variance = float(np.var(image_gray))

    # Step 3: Compute global gradient magnitude
    grad_x = sobel(image_gray, axis=1)  # Horizontal gradient
    grad_y = sobel(image_gray, axis=0)  # Vertical gradient
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    global_gradient_magnitude = np.mean(gradient_magnitude)

    return (height, width), global_variance, global_gradient_magnitude

def get_kernel_patch_sizes(image):
    """
    Select kernel and patch size based on the image's global variance and resolution.

    Args:
        image (numpy.ndarray): The input image as a 2D array (grayscale).

    Returns:
        dict: A dictionary containing:
            - 'kernel_size': The selected kernel size as a tuple (height, width).
            - 'patch_size': The selected patch size as a tuple (height, width).
            - 'global_variance': The global variance of the image.
            - 'resolution': The resolution of the image.
    """
    # Compute global properties of the image
    resolution, global_variance, _ = get_global_properties(image)

    # Determine kernel size based on global variance
    if global_variance < 2000:  # Low variance
        kernel_size = (5, 5)
    elif global_variance < 3000:  # Medium variance
        kernel_size = (3, 3)
    else:  # High variance
        kernel_size = (3, 3)

    # Determine patch size based on resolution
    if resolution[0] >= 1024 or resolution[1] >= 1024:  # Very high resolution
        patch_size = (128, 128)
    elif resolution[0] >= 512 and resolution[1] >= 512:  # High resolution
        patch_size = (64, 64)
    else:  # Low resolution
        patch_size = (32, 32)

    return {
        "kernel_size": kernel_size,
        "patch_size": patch_size,
        "global_variance": global_variance,
        "resolution": resolution,
    }

def divide_into_patches(image, overlap_percentage=50):
    """
    Divides the input image into patches using a sliding window approach with overlap.
    
    Args:
        image (numpy.ndarray): The input image as a 2D array (grayscale).
        overlap_percentage (int, optional): The percentage of overlap between patches. Default is 50%.

    Returns:
        list: A list of patches extracted from the image.
    """
    # Get the kernel and patch sizes using previously defined functions
    properties = get_kernel_patch_sizes(image)
    patch_size = properties['patch_size']

    # Calculate step size based on overlap percentage
    step_size = int((1 - overlap_percentage / 100) * patch_size[0])  # calculate step size based on overlap

    patches = []
    height, width = image.shape

    for i in range(0, height - patch_size[0] + 1, step_size):
        for j in range(0, width - patch_size[1] + 1, step_size):
            patch = image[i:i + patch_size[0], j:j + patch_size[1]]
            patches.append(patch)

    return patches