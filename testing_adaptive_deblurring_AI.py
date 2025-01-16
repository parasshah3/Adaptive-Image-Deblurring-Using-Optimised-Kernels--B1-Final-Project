import numpy as np
import matplotlib.pyplot as plt
from adaptive_deblurring_functions_AI import load_image, add_noise, reconstruct_image_extended

# Load a sharp reference image
sharp_image_path = "/Users/paras/Desktop/B1 Final Project/cameraman.tif"  # Replace with the correct path
sharp_image = load_image(sharp_image_path)

# Simulate a noisy input image (e.g., Gaussian noise)
noisy_image = add_noise(sharp_image, noise_type="gaussian", mean=0, var=0.05)

# Parameters for the reconstruction process
gaussian_variance = 1.0  # Variance for Gaussian blur kernel
high_scaling_factor = 1.81  # High-pass filter scaling factor
brightness_factor = 2.01  # Brightness adjustment factor
edge_scaling_factor = 1.33  # Edge enhancement scaling factor (new parameter)
overlap_percentage = 90  # Overlap between patches

# Perform reconstruction
reconstructed_image = reconstruct_image_extended(
    input_image=noisy_image,
    gaussian_variance=gaussian_variance,
    high_scaling_factor=high_scaling_factor,
    brightness_factor=brightness_factor,
    edge_scaling_factor=edge_scaling_factor,  # Pass the edge scaling factor
    overlap_percentage=overlap_percentage
)

# Plot the sharp image, noisy input image, and reconstructed image
plt.figure(figsize=(15, 5))

# Sharp reference image
plt.subplot(1, 3, 1)
plt.imshow(sharp_image, cmap="gray")
plt.title("Sharp Reference Image")
plt.axis("off")

# Noisy input image
plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap="gray")
plt.title("Noisy Input Image")
plt.axis("off")

# Reconstructed image
plt.subplot(1, 3, 3)
plt.imshow(reconstructed_image, cmap="gray")
plt.title("AI Reconstructed Image")
plt.axis("off")

# Display the results
plt.tight_layout()
plt.show()