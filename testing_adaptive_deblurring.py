import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from adaptive_deblurring_functions import load_image, divide_into_patches

# Load an image (choose one from the provided paths)
image_path = "/Users/paras/Desktop/B1 Final Project/cameraman.tif"
image = load_image(image_path, grayscale=True)

# Divide the image into patches using a 50% overlap
patches_list = divide_into_patches(image, overlap_percentage=50)

# Define the color list to differentiate patches
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'purple', 'brown', 'pink']

# Plot the image
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image, cmap='gray')

# Get the image dimensions and patch size
height, width = image.shape
patch_height, patch_width = patches_list[0].shape  # All patches have the same size

# Calculate center position
center_x = width // 2
center_y = height // 2

# Calculate patch offsets to center 3 consecutive patches
start_patch_idx = len(patches_list) // 2  # Start from the middle of the patches list
patch_spacing = patch_width // 2  # Overlap is 50%, so the spacing is half the patch width

# List to hold labels for legend
patch_labels = []

# Plot three consecutive patches centered around the middle of the image
for i, patch_idx in enumerate([start_patch_idx, start_patch_idx + 1, start_patch_idx + 2]):
    patch = patches_list[patch_idx]
    
    # Calculate the top-left corner of the patch to place it near the center
    top_left_x = center_x - patch_width // 2 + i * patch_spacing  # Adjust x-coordinate for spacing
    top_left_y = center_y - patch_height // 2  # Adjust y-coordinate for centering

    # Use different colors for each rectangle
    rect = patches.Rectangle((top_left_x, top_left_y), patch.shape[1], patch.shape[0], linewidth=2,
                             edgecolor=colors[i % len(colors)], facecolor='none')

    ax.add_patch(rect)
    
    # Add the label to the patch_labels list for the legend
    patch_labels.append(f'Patch {patch_idx + 1}')

# Add legend outside the image
plt.legend(patch_labels, loc='center left', bbox_to_anchor=(1, 0.5), title='Patches')

# Display the image with rectangles and labels
plt.title("Three Consecutive Patches Centered in the Image")
plt.tight_layout()
plt.show()