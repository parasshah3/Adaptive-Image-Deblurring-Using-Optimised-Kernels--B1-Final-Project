import numpy as np
import matplotlib.pyplot as plt
from adaptive_deblurring_functions import (
    load_image,
    divide_into_patches,
    get_kernel_patch_sizes,
    dynamic_kernel_selection,
    image_reconstruction,
)

# Load the image
image_path = "/Users/paras/Desktop/B1 Final Project/cameraman.tif"
image = load_image(image_path)

# Divide into patches
overlap_percentage = 75  # Define the overlap percentage
patches = divide_into_patches(image, overlap_percentage)

# Get global properties to determine patch and kernel sizes
global_properties = get_kernel_patch_sizes(image)
patch_size = global_properties["patch_size"]
kernel_size = global_properties["kernel_size"]

# Select kernels dynamically and get their variance types
kernels, variance_types = dynamic_kernel_selection(patches, kernel_size)

# Reconstruct the image
reconstructed_image = image_reconstruction(
    image_shape=image.shape,
    patches=patches,
    kernels=kernels,
    variance_types=variance_types,
    patch_size=patch_size,
    overlap_percentage=overlap_percentage,
)

# Display the images
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Reconstructed Image
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap="gray")
plt.title("Reconstructed Image with Kernel Regularisation") 
plt.axis("off")

# Show the results
plt.tight_layout()
plt.show()