import time
import numpy as np
import matplotlib.pyplot as plt
from adaptive_deblurring_functions import (
    load_image,
    add_noise,
    optimise_parameters,
    reconstruct_image,
)
from adaptive_deblurring_functions_AI import (
    gradient_descent_optimisation,
    reconstruct_image_extended,
)

# Path to the sharp (reference) image
sharp_image_path = "/Users/paras/Desktop/B1 Final Project/pirate.tif"

# Load the sharp reference image
sharp_reference = load_image(sharp_image_path)

# Simulate a noisy (blurry) input image
noise_type = "speckle"  # Options: "gaussian", "salt_and_pepper", "poisson", "speckle"
mean, variance = 0, 0.01
blurry_image = add_noise(sharp_reference, noise_type=noise_type, mean=mean, var=variance)

# Parameters
gaussian_variance = 1.0
overlap_percentage = 75

# Time the Original Two-Kernel Method
start_time_original = time.time()
original_results = optimise_parameters(
    input_image=blurry_image,
    reference_image=sharp_reference,
    gaussian_variance=gaussian_variance,
    overlap_percentage=overlap_percentage,
)
original_time = time.time() - start_time_original

# Reconstruct the image using the optimised parameters (Original Method)
reconstructed_original = reconstruct_image(
    input_image=blurry_image,
    gaussian_variance=gaussian_variance,
    high_scaling_factor=original_results["optimised_params"][1],
    brightness_factor=original_results["optimised_params"][0],
    overlap_percentage=overlap_percentage,
)

# Time the Extended Method
start_time_extended = time.time()
extended_results = gradient_descent_optimisation(
    input_image=blurry_image,
    reference_image=sharp_reference,
    gaussian_variance=gaussian_variance,
    overlap_percentage=overlap_percentage,
    initial_guess=[1.01, 1.11, 0.65],
    learning_rate=0.001,
    tolerance=1e-3,
    max_iterations=50,
)
extended_time = time.time() - start_time_extended

# Reconstruct the image using the optimised parameters (Extended Method)
reconstructed_extended = reconstruct_image_extended(
    input_image=blurry_image,
    gaussian_variance=gaussian_variance,
    high_scaling_factor=extended_results["optimised_params"][1],
    brightness_factor=extended_results["optimised_params"][0],
    edge_scaling_factor=extended_results["optimised_params"][2],
    overlap_percentage=overlap_percentage,
)

# Print Timing Results
print("Timing Results:")
print(f"Original Two-Kernel Method Time: {original_time:.2f} seconds")
print(f"Extended Method Time: {extended_time:.2f} seconds")

# Plot Results
plt.figure(figsize=(15, 5))

# Blurry Image
plt.subplot(1, 3, 1)
plt.imshow(blurry_image, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

# Original Method Reconstruction
plt.subplot(1, 3, 2)
plt.imshow(reconstructed_original, cmap="gray")
plt.title(f"Original Method (Time: {original_time:.2f}s)")
plt.axis("off")

# Extended Method Reconstruction
plt.subplot(1, 3, 3)
plt.imshow(reconstructed_extended, cmap="gray")
plt.title(f"Extended Method (Time: {extended_time:.2f}s)")
plt.axis("off")

plt.tight_layout()
plt.show()