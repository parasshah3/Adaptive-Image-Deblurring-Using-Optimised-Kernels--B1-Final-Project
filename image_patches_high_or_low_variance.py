import matplotlib.pyplot as plt
import numpy as np
from adaptive_deblurring_functions import (
    load_image,
    divide_into_patches,
    get_kernel_patch_sizes,
    dynamic_kernel_selection
)

def highlight_consistent_overlapping_regions(image, patches, kernels, patch_size, overlap_percentage=50):
    """
    Highlight overlapping regions of patches in the image, colour-coding them based on the kernel type:
    - High variance (sharpening kernel): Red colour.
    - Low variance (Gaussian blur kernel): Green colour.
    Overlapping regions with mixed variance are not highlighted.

    Args:
        image (numpy.ndarray): The input grayscale image.
        patches (list of numpy.ndarray): List of patches extracted from the image.
        kernels (list of numpy.ndarray): List of kernels corresponding to each patch.
        patch_size (tuple): The size of each patch (height, width).
        overlap_percentage (float): Overlap percentage between patches.

    Returns:
        numpy.ndarray: A coloured map highlighting consistent overlapping regions of the patches.
    """
    # Create a coloured map for visualization (convert grayscale to RGB)
    coloured_map = np.stack([image, image, image], axis=-1).astype(np.uint8)

    # Define colours
    high_variance_colour = [255, 0, 0]  # Red for high variance (sharpening kernel)
    low_variance_colour = [0, 255, 0]   # Green for low variance (Gaussian blur kernel)

    # Calculate step size based on overlap
    step_size = int((1 - overlap_percentage / 100) * patch_size[0])

    # Initialize tracking maps for high and low variance overlaps
    high_variance_map = np.zeros_like(image, dtype=int)
    low_variance_map = np.zeros_like(image, dtype=int)

    # Image dimensions
    height, width = image.shape
    patch_index = 0

    # Iterate over patches and classify based on kernels
    for i in range(0, height - patch_size[0] + 1, step_size):
        for j in range(0, width - patch_size[1] + 1, step_size):
            if patch_index >= len(patches) or patch_index >= len(kernels):
                break

            kernel = kernels[patch_index]

            # Determine if the kernel is a sharpening kernel (high variance)
            if np.sum(kernel) > 1:  # Sharpening kernel
                high_variance_map[i:i + patch_size[0], j:j + patch_size[1]] += 1
            else:  # Gaussian blur kernel
                low_variance_map[i:i + patch_size[0], j:j + patch_size[1]] += 1

            patch_index += 1

    # Identify consistent overlapping regions
    high_variance_overlap = (high_variance_map > 1) & (low_variance_map == 0)
    low_variance_overlap = (low_variance_map > 1) & (high_variance_map == 0)

    # Colour the consistent overlapping regions
    coloured_map[high_variance_overlap] = high_variance_colour
    coloured_map[low_variance_overlap] = low_variance_colour

    return coloured_map


# Load the input image
image_path = '/Users/paras/Desktop/B1 Final Project/cameraman.tif'  # Replace with your image path
image = load_image(image_path, grayscale=True)

# Get global properties
global_properties = get_kernel_patch_sizes(image)
patch_size = global_properties['patch_size']
kernel_size = global_properties['kernel_size']

# Divide the image into patches
overlap_percentage = 50  # Define the overlap percentage
patches = divide_into_patches(image, overlap_percentage)

# Generate kernels for each patch
kernels = dynamic_kernel_selection(patches, kernel_size)

# Highlight consistent overlapping regions
coloured_map = highlight_consistent_overlapping_regions(image, patches, kernels, patch_size, overlap_percentage)

# Display the original image and the highlighted patch map
plt.figure(figsize=(14, 8))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Coloured Map with Overlaps
plt.subplot(1, 2, 2)
plt.imshow(coloured_map)
plt.title("Consistent Overlapping Regions Highlighted")
plt.axis('off')

# Add a key for high and low variance
plt.figtext(0.5, 0.01, "Key: Red = High Variance Overlap, Green = Low Variance Overlap, Mixed = Not Highlighted", 
            ha='center', color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()