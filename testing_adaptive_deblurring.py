import numpy as np
import matplotlib.pyplot as plt
from adaptive_deblurring_functions import (
    load_image,
    divide_into_patches,
    get_kernel_patch_sizes,
    dynamic_kernel_selection,
    image_reconstruction,
    get_low_to_high_variance_threshold,
)

# Load the image
image_path = "/Users/paras/Desktop/B1 Final Project/cameraman.tif"
image = load_image(image_path)

# Divide into patches
overlap_percentage = 90  # Define the overlap percentage
patches = divide_into_patches(image, overlap_percentage)

# Get global properties to determine patch and kernel sizes
global_properties = get_kernel_patch_sizes(image)
print(f"global_properties: {global_properties}")
patch_size = global_properties["patch_size"]
kernel_size = global_properties["kernel_size"]

# Get high to low variance threshold
threshold = get_low_to_high_variance_threshold(patches)
print(f"Variance threshold: {threshold}")

# Choose scaling factor and Gaussian variance
scaling_factor = 1.6  # Adjust as needed
gaussian_variance = 50 # Adjust as needed

# Select kernels dynamically and get their variance types
kernels, variance_types = dynamic_kernel_selection(patches, kernel_size, threshold, scaling_factor, gaussian_variance)

# Reconstruct the image
reconstructed_image = image_reconstruction(
    image_shape=image.shape,
    patches=patches,
    kernels=kernels,
    variance_types=variance_types,
    patch_size=patch_size,
    overlap_percentage=overlap_percentage,
    high_scaling_factor=scaling_factor,
    gaussian_variance=gaussian_variance,
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