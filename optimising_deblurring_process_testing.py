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
image_path = "/Users/paras/Desktop/B1 Final Project/defocused building.JPG"
image = load_image(image_path)

# Divide into patches
overlap_percentage = 90  # Define the overlap percentage
patches = divide_into_patches(image, overlap_percentage)

# Get global properties to determine patch and kernel sizes
global_properties = get_kernel_patch_sizes(image)
patch_size = global_properties["patch_size"]
kernel_size = global_properties["kernel_size"]

# Get high to low variance threshold
threshold = get_low_to_high_variance_threshold(patches)

# Fixed Gaussian variance and scaling factor
gaussian_variance = 1  # Adjust as needed
scaling_factor = 1.6  # Adjust as needed

# Range of brightness factor values to iterate through
brightness_values = np.linspace(0.31, 0.61, 5)  # Adjust range and number of values as needed

# Plot setup
fig, axes = plt.subplots(1, len(brightness_values), figsize=(15, 5))
fig.suptitle(f"Effect of Brightness Factor on Image Reconstruction (Gaussian Variance: {gaussian_variance})", fontsize=16)

# Iterate through brightness factor values
for idx, brightness_factor in enumerate(brightness_values):
    # Select kernels dynamically and get their variance types
    kernels, variance_types = dynamic_kernel_selection(
        patches, kernel_size, threshold, scaling_factor, gaussian_variance, brightness_factor
    )

    # Reconstruct the image with varying brightness
    reconstructed_image = image_reconstruction(
        image_shape=image.shape,
        patches=patches,
        kernels=kernels,
        variance_types=variance_types,
        patch_size=patch_size,
        overlap_percentage=overlap_percentage,
        high_scaling_factor=scaling_factor,
        gaussian_variance=gaussian_variance,
        brightness_factor=brightness_factor,  # Pass brightness factor here
    )

    # Plot the reconstructed image
    ax = axes[idx]
    ax.imshow(reconstructed_image, cmap="gray")
    ax.set_title(f"B: {brightness_factor:.2f}", fontsize=10)
    ax.axis("off")

# Adjust layout and display the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()