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

from scipy.ndimage import sobel

def get_properties(image):
    """
    Computes the properties of the image or patch: resolution, variance, and gradient magnitude.

    Args:
        image (numpy.ndarray): The input image or patch as a 2D array (grayscale).

    Returns:
        tuple: A tuple containing:
            - resolution (tuple): Tuple (height, width) of the image/patch.
            - variance (float): Variance of the image/patch intensity.
            - gradient_magnitude (float): Average gradient magnitude of the image/patch.
    """
    # Step 1: Compute resolution (height, width)
    height, width = image.shape

    # Step 2: Compute variance
    # If the image has multiple channels (RGB), convert to grayscale by averaging channels
    if len(image.shape) == 3:  # RGB Image
        image_gray = np.mean(image, axis=2)  # Convert to grayscale by averaging channels
    else:
        image_gray = image  # Grayscale Image or 2D patch

    # Compute variance
    variance = float(np.var(image_gray))

    # Step 3: Compute gradient magnitude
    grad_x = sobel(image_gray, axis=1)  # Horizontal gradient
    grad_y = sobel(image_gray, axis=0)  # Vertical gradient
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = np.mean(gradient_magnitude)

    return (height, width), variance, gradient_magnitude 

def kernel_library():
    """
    Return a dictionary of predefined kernels for various sizes and variance types.

    Returns:
        dict: A nested dictionary with variance types ('high', 'low') and kernel sizes as keys.
    """
    return {
        (3, 3): {
            "high": np.array([[-1, -1, -1], 
                              [-1, 16, -1], 
                              [-1, -1, -1]]),
            "low": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
        },
        (5, 5): {
            "high": np.array([[0, 0, 1, 0, 0], 
                              [0, 1, 1, 1, 0], 
                              [1, 1, -4, 1, 1], 
                              [0, 1, 1, 1, 0], 
                              [0, 0, 1, 0, 0]]),
            "low": np.array([[1, 4, 6, 4, 1], 
                             [4, 16, 24, 16, 4], 
                             [6, 24, 36, 24, 6], 
                             [4, 16, 24, 16, 4], 
                             [1, 4, 6, 4, 1]]) / 256,
        },
        (7, 7): {
            "high": np.array([[0, 0, 0, 1, 0, 0, 0], 
                              [0, 1, 1, 2, 1, 1, 0], 
                              [0, 1, 1, 4, 1, 1, 0], 
                              [1, 2, 4, -12, 4, 2, 1], 
                              [0, 1, 1, 4, 1, 1, 0], 
                              [0, 1, 1, 2, 1, 1, 0], 
                              [0, 0, 0, 1, 0, 0, 0]]),
            "low": np.array([[1, 6, 15, 20, 15, 6, 1], 
                             [6, 36, 90, 120, 90, 36, 6], 
                             [15, 90, 225, 300, 225, 90, 15],
                             [20, 120, 300, 400, 300, 120, 20], 
                             [15, 90, 225, 300, 225, 90, 15], 
                             [6, 36, 90, 120, 90, 36, 6], 
                             [1, 6, 15, 20, 15, 6, 1]]) / 1600,
        },
    }

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
    resolution, global_variance, _ = get_properties(image)

    # Determine kernel size based on global variance
    if global_variance < 1000:  # Low variance
        kernel_size = (7, 7)
    elif global_variance < 2000:  # Medium variance
        kernel_size = (5, 5)
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
    print(f"Step size: {step_size}")

    patches = []
    height, width = image.shape

    # Calculate expected number of patches
    expected_num_patches_x = (width - patch_size[1]) // step_size + 1
    expected_num_patches_y = (height - patch_size[0]) // step_size + 1
    expected_num_patches = expected_num_patches_x * expected_num_patches_y

    # Print expected number of patches for verification
    print(f"Expected number of patches: {expected_num_patches}")

    # Iterate over the image to extract patches
    for i in range(0, height - patch_size[0] + 1, step_size):
        for j in range(0, width - patch_size[1] + 1, step_size):
            patch = image[i:i + patch_size[0], j:j + patch_size[1]]
            patches.append(patch)

    # Verify the actual number of patches matches the expected
    actual_num_patches = len(patches)
    if actual_num_patches == expected_num_patches:
        print(f"Patch count verification passed! Actual patches: {actual_num_patches}, Expected: {expected_num_patches}")
    else:
        print(f"Patch count verification failed! Actual patches: {actual_num_patches}, Expected: {expected_num_patches}")

    return patches

import numpy as np

def dynamic_local_kernel_starting_point(patch, kernel_size):
    """
    Return a filter (kernel) starting point for a given patch based on its local variance
    and the specified kernel size.

    Args:
        patch (numpy.ndarray): The patch of the image (grayscale).
        kernel_size (tuple): The kernel size as a tuple (height, width) (fixed size).

    Returns:
        tuple:
            - kernel (numpy.ndarray): A filter (kernel) starting point for high or low local variance.
            - variance_type (str): "high" or "low" indicating variance type.
    """
    kernels = kernel_library()  # Fetch kernel dictionary
    _, local_variance, _ = get_properties(patch)  # Get variance

    variance_type = "high" if local_variance > 3000 else "low"
    kernel = kernels[kernel_size][variance_type]

    return kernel, variance_type

def dynamic_kernel_selection(patches, kernel_size):
    """
    Select kernels dynamically for each patch and return both the kernel and variance type.

    Args:
        patches (list of numpy.ndarray): List of 2D numpy arrays representing the image patches.
        kernel_size (tuple): The kernel size as a tuple (height, width).

    Returns:
        tuple:
            - kernels (list of numpy.ndarray): List of 2D kernels for each patch.
            - variance_types (list of str): List indicating "high" or "low" variance for each patch.
    """
    kernels = []
    variance_types = []

    # Iterate over each patch
    for patch in patches:
        # Use dynamic_local_kernel_starting_point to determine kernel and variance type
        kernel, variance_type = dynamic_local_kernel_starting_point(patch, kernel_size)

        # Append the kernel and variance type to the respective lists
        kernels.append(kernel)
        variance_types.append(variance_type)

    return kernels, variance_types

import numpy as np
from scipy.signal import convolve2d

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

from scipy.signal import convolve2d
import numpy as np

import numpy as np
from scipy.signal import convolve2d

def image_reconstruction_og(image_shape, patches, kernels, variance_types, patch_size, overlap_percentage=50):
    """
    Reconstruct the deblurred image by applying kernels to patches and blending mixed regions.

    Args:
        image_shape (tuple): Shape of the original image (height, width).
        patches (list of numpy.ndarray): List of 2D numpy arrays representing the patches.
        kernels (list of numpy.ndarray): List of 2D kernels for each patch.
        variance_types (list of str): List of variance types ("high" or "low") for each patch.
        patch_size (tuple): Tuple (height, width) representing the size of each patch.
        overlap_percentage (float): Overlap percentage between patches (default: 50%).

    Returns:
        numpy.ndarray: The reconstructed deblurred image.
    """
    output_image = np.zeros(image_shape, dtype=np.float32)
    patch_counts = np.zeros(image_shape, dtype=np.float32)
    step_size = int((1 - overlap_percentage / 100) * patch_size[0])

    kernels_dict = kernel_library()  # Fetch kernel dictionary

    patch_index = 0
    for i in range(0, image_shape[0] - patch_size[0] + 1, step_size):
        for j in range(0, image_shape[1] - patch_size[1] + 1, step_size):
            patch = patches[patch_index]
            kernel = kernels[patch_index]
            variance_type = variance_types[patch_index]

            # Use kernel.shape to determine the kernel size
            kernel_size = kernel.shape

            overlapping_high = patch_counts[i:i + patch_size[0], j:j + patch_size[1]] > 0

            if np.any(overlapping_high):
                other_type = "low" if variance_type == "high" and patch_index != 0 else "high"
                regularised_kernel = (kernel + kernels_dict[kernel_size][other_type]) / 2
                kernel = regularised_kernel

            convolved_patch = convolve2d(patch, kernel, mode="same", boundary="symm")
            output_image[i:i + patch_size[0], j:j + patch_size[1]] += convolved_patch
            patch_counts[i:i + patch_size[0], j:j + patch_size[1]] += 1
            patch_index += 1

    return np.divide(output_image, patch_counts, where=patch_counts != 0)

def image_reconstruction(image_shape, patches, kernels, variance_types, patch_size, overlap_percentage=50):
    """
    Reconstruct the deblurred image by applying kernels to patches and blending mixed regions.

    Args:
        image_shape (tuple): Shape of the original image (height, width).
        patches (list of numpy.ndarray): List of 2D numpy arrays representing the patches.
        kernels (list of numpy.ndarray): List of 2D kernels for each patch.
        variance_types (list of str): List of variance types ("high" or "low") for each patch.
        patch_size (tuple): Tuple (height, width) representing the size of each patch.
        overlap_percentage (float): Overlap percentage between patches (default: 50%).

    Returns:
        numpy.ndarray: The reconstructed deblurred image.
    """
    output_image = np.zeros(image_shape, dtype=np.float32)
    patch_counts = np.zeros(image_shape, dtype=np.float32)
    step_size = int((1 - overlap_percentage / 100) * patch_size[0])

    kernels_dict = kernel_library()  # Fetch kernel dictionary

    patch_index = 0
    for i in range(0, image_shape[0] - patch_size[0] + 1, step_size):
        for j in range(0, image_shape[1] - patch_size[1] + 1, step_size):
            patch = patches[patch_index]
            kernel = kernels[patch_index]
            variance_type = variance_types[patch_index]

            # Skip kernel regularisation for the very first patch (top-left corner)
            if patch_index == 0: 
                print(f"Skipping kernel regularisation for the very first patch: patch_index={patch_index} and setting kernel manually.")
                print(f"Kernel: {kernel}")
                kernel = (kernel + kernels_dict[kernel.shape]['high']) / 2
            else:
                # Check for overlap with a different variance type
                overlapping_high = patch_counts[i:i + patch_size[0], j:j + patch_size[1]] > 0

                if np.any(overlapping_high):
                    other_type = "low" if variance_type == "high" else "high"
                    regularised_kernel = (kernel + kernels_dict[kernel.shape][other_type]) / 2
                    kernel = regularised_kernel

            # Convolve the patch with its kernel
            convolved_patch = convolve2d(patch, kernel, mode="same", boundary="symm")
            output_image[i:i + patch_size[0], j:j + patch_size[1]] += convolved_patch
            patch_counts[i:i + patch_size[0], j:j + patch_size[1]] += 1
            patch_index += 1

    return np.divide(output_image, patch_counts, where=patch_counts != 0)