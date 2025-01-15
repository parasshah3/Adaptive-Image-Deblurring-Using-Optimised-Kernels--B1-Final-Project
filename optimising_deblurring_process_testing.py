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
image_path = "/Users/paras/Desktop/B1 Final Project/sharp_building.JPG"
image = load_image(image_path)

# Divide into patches
overlap_percentage = 75  # Define the overlap percentage
patches = divide_into_patches(image, overlap_percentage)

# Get global properties to determine patch and kernel sizes
global_properties = get_kernel_patch_sizes(image)
patch_size = global_properties["patch_size"]
kernel_size = global_properties["kernel_size"]

# Get high to low variance threshold
threshold = get_low_to_high_variance_threshold(patches)

# Define ranges for Gaussian variance and high scaling factor
gaussian_variance_values = np.linspace(0.5, 3.0, 5)  # Adjust range and number of values as needed
scaling_factor_values = np.linspace(1.0, 5.0, 5)  # Adjust range and number of values as needed

# Plot setup
fig, axes = plt.subplots(len(gaussian_variance_values), len(scaling_factor_values), figsize=(15, 12))
fig.suptitle("Effect of Gaussian Variance and High Scaling Factor on Image Reconstruction", fontsize=16)

# Iterate through combinations of Gaussian variance and scaling factor
for i, gaussian_variance in enumerate(gaussian_variance_values):
    for j, scaling_factor in enumerate(scaling_factor_values):
        # Select kernels dynamically and get their variance types
        kernels, variance_types = dynamic_kernel_selection(
            patches, kernel_size, threshold, scaling_factor, gaussian_variance
        )

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

        # Plot the reconstructed image
        ax = axes[i, j]
        ax.imshow(reconstructed_image, cmap="gray")
        ax.set_title(f"G: {gaussian_variance:.2f}, S: {scaling_factor:.2f}", fontsize=10)
        ax.axis("off")

# Adjust layout and display the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()