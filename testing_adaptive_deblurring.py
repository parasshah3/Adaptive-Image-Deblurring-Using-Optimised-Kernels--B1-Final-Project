import matplotlib.pyplot as plt
from adaptive_deblurring_functions import (
    load_image,
    divide_into_patches,
    get_kernel_patch_sizes,
    dynamic_kernel_selection,
    image_reconstruction,
)

# Step 1: Load the input image
image_path = '/Users/paras/Desktop/B1 Final Project/cameraman.tif'  # Replace with the actual path to your image
input_image = load_image(image_path, grayscale=True)

# Step 2: Divide the image into patches
overlap_percentage = 50  # Define the overlap percentage
patches = divide_into_patches(input_image, overlap_percentage=overlap_percentage)

# Step 3: Get global properties and determine kernel size
global_properties = get_kernel_patch_sizes(input_image)
kernel_size = global_properties['kernel_size']
patch_size = global_properties['patch_size']
print(f"patch_size: {patch_size}, kernel_size: {kernel_size}")

# Step 4: Generate dynamic kernels for each patch
kernels = dynamic_kernel_selection(patches, kernel_size)

# Step 5: Reconstruct the image using the patches and kernels
print(f"patch_size: {patch_size} (type: {type(patch_size)})")
if not isinstance(patch_size, tuple):
    raise TypeError(f"patch_size must be a tuple, but got {type(patch_size)}: {patch_size}")
reconstructed_image = image_reconstruction(input_image.shape, patches, kernels, patch_size, overlap_percentage)

# Step 6: Display the results
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Reconstructed Image
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Deblurred Image using Optimized Kernels')
plt.axis('off')

plt.show()