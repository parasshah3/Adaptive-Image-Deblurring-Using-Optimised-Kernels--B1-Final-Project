import numpy as np
import matplotlib.pyplot as plt
from adaptive_deblurring_functions import load_image, get_properties, dynamic_local_kernel_starting_point, divide_into_patches

# Load the input image (cameraman image)
image_path = "/Users/paras/Desktop/B1 Final Project/cameraman.tif"
image = load_image(image_path, grayscale=True)

# Define overlap percentage (e.g., 50%)
overlap_percentage = 50

# Divide the image into patches
patches = divide_into_patches(image, overlap_percentage=overlap_percentage)

# Initialize a list to store local variances
local_variances = []

# Compute local variance for each patch and store it
for i, patch in enumerate(patches):
    _, local_variance, _ = get_properties(patch)
    local_variances.append(local_variance)

# Calculate range, mean, and median of local variances
variance_range = np.ptp(local_variances)  # Range = max - min
variance_mean = np.mean(local_variances)
variance_median = np.median(local_variances)

# Display the range, mean, and median of the local variances
print(f"Range of Local Variances: {variance_range:.4f}")
print(f"Mean of Local Variances: {variance_mean:.4f}")
print(f"Median of Local Variances: {variance_median:.4f}")

# Now let's also visualize which kernel has been chosen for each patch
for i, patch in enumerate(patches):
    # Get the kernel for the current patch
    _, local_variance, _ = get_properties(patch)
    
    # Determine which kernel to use for the current patch
    kernel = dynamic_local_kernel_starting_point(patch, kernel_size=(3, 3))  # Example for kernel size 3x3

    # Display the selected kernel
    print(f"Patch {i+1}: Local Variance = {local_variance:.4f}, Selected Kernel:")
    print(kernel)
    
    # Visualize patch and draw a red rectangle around it
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.gca().add_patch(plt.Rectangle((i * 64, 0), 64, 64, linewidth=2, edgecolor='r', facecolor='none'))
    plt.title(f"Patch {i+1} - Local Variance: {local_variance:.4f}")
    plt.show()