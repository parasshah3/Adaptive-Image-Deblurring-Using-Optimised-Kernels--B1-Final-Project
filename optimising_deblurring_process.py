import matplotlib.pyplot as plt
from adaptive_deblurring_functions import load_image, reconstruct_image

# Load the image
image_path = "/Users/paras/Desktop/B1 Final Project/cameraman.tif"  # Update with your image path
input_image = load_image(image_path)

# Define parameters
gaussian_variance = 1.0       # Variance for the Gaussian blur kernel
high_scaling_factor = 1.55     # Scaling factor for high-pass kernels
brightness_factor = 0.35       # Brightness adjustment for low-pass kernels
overlap_percentage = 90       # Overlap percentage between patches

# Reconstruct the image
reconstructed_image = reconstruct_image(
    input_image=input_image,
    gaussian_variance=gaussian_variance,
    high_scaling_factor=high_scaling_factor,
    brightness_factor=brightness_factor,
    overlap_percentage=overlap_percentage
)

# Plot the original and reconstructed images
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Reconstructed Image
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap="gray")
plt.title("Reconstructed Image")
plt.axis("off")

plt.tight_layout()
plt.show()