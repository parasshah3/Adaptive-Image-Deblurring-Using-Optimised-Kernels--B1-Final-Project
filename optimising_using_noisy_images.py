import numpy as np
import matplotlib.pyplot as plt
from adaptive_deblurring_functions import load_image, optimise_parameters, add_noise, reconstruct_image

# Path to the sharp (reference) image
sharp_image_path = "/Users/paras/Desktop/B1 Final Project/pirate.tif"

# Load the sharp reference image
sharp_reference = load_image(sharp_image_path)

# Add noise to the sharp image to simulate a blurry image
noise_type = "poisson"  # Choose from "gaussian", "salt_and_pepper", "poisson", or "speckle"
mean = 0  # Mean for Gaussian noise
variance = 0.01  # Variance for Gaussian noise
blurry_image = add_noise(sharp_reference, noise_type=noise_type, mean=mean, var=variance)

# Parameters for Gaussian variance and overlap percentage
gaussian_variance = 1.0  # Adjust as needed
overlap_percentage = 75  # Overlap percentage for patches

# Perform optimisation
optimisation_results = optimise_parameters(
    input_image=blurry_image,
    reference_image=sharp_reference,
    gaussian_variance=gaussian_variance,
    overlap_percentage=overlap_percentage,
)

# Display optimisation results
print("Optimisation Results:")
print(f"Optimised Brightness Factor: {optimisation_results['optimised_params'][0]:.4f}")
print(f"Optimised High Scaling Factor: {optimisation_results['optimised_params'][1]:.4f}")
print(f"Minimum MSE: {optimisation_results['minimum_mse']:.4f}")
print(f"Optimisation Success: {optimisation_results['success']}")
print(f"Message: {optimisation_results['message']}")

# Extract optimised parameters
optimised_brightness = optimisation_results['optimised_params'][0]
optimised_scaling = optimisation_results['optimised_params'][1]

# Reconstruct the image using optimised parameters
reconstructed_image = reconstruct_image(
    input_image=blurry_image,
    gaussian_variance=gaussian_variance,
    high_scaling_factor=optimised_scaling,
    brightness_factor=optimised_brightness,
    overlap_percentage=overlap_percentage,
)

# Plot the noisy image, reference image, and optimised reconstructed image
plt.figure(figsize=(15, 5))

# Noisy (blurry) image
plt.subplot(1, 3, 1)
plt.imshow(blurry_image, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

# Reference sharp image
plt.subplot(1, 3, 2)
plt.imshow(sharp_reference, cmap="gray")
plt.title("Reference Sharp Image")
plt.axis("off")

# Reconstructed image with optimised parameters
plt.subplot(1, 3, 3)
plt.imshow(reconstructed_image, cmap="gray")
plt.title("Reconstructed Image")
plt.axis("off")

# Show plots
plt.tight_layout()
plt.show()