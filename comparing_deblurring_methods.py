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
sharp_image_path = "/Users/paras/Desktop/B1 Final Project/cameraman.tif"

# Load the sharp reference image
sharp_reference = load_image(sharp_image_path)

# Simulate a noisy image
noise_type = "poisson"  # Choose from "gaussian", "salt_and_pepper", "poisson", "speckle"
mean = 0
variance = 0.01
blurry_image = add_noise(sharp_reference, noise_type=noise_type, mean=mean, var=variance)

# Parameters for both methods
gaussian_variance = 1.0
overlap_percentage = 75

# ----------- Method 1: Original Two-Kernel Method -----------
print("Optimising Original Two-Kernel Method...")
original_results = optimise_parameters(
    input_image=blurry_image,
    reference_image=sharp_reference,
    gaussian_variance=gaussian_variance,
    overlap_percentage=overlap_percentage,
)
optimised_brightness_1 = original_results["optimised_params"][0]
optimised_scaling_1 = original_results["optimised_params"][1]
reconstructed_image_1 = reconstruct_image(
    input_image=blurry_image,
    gaussian_variance=gaussian_variance,
    high_scaling_factor=optimised_scaling_1,
    brightness_factor=optimised_brightness_1,
    overlap_percentage=overlap_percentage,
)

# Display results for Method 1
print("\nOriginal Two-Kernel Method Results:")
print(f"Optimised Brightness Factor: {optimised_brightness_1:.4f}")
print(f"Optimised High Scaling Factor: {optimised_scaling_1:.4f}")
print(f"Minimum MSE: {original_results['minimum_mse']:.4f}")
print(f"Optimisation Success: {original_results['success']}")
print(f"Message: {original_results['message']}")

# ----------- Method 2: Extended Multi-Kernel Method -----------
print("\nOptimising Extended Four-Kernel Method...")
initial_guess = [0.4, 1.7, 0.8]  # Initial guesses for [brightness_factor, high_scaling_factor, edge_scaling_factor]
extended_results = gradient_descent_optimisation(
    input_image=blurry_image,
    reference_image=sharp_reference,
    gaussian_variance=gaussian_variance,
    overlap_percentage=overlap_percentage,
    initial_guess=initial_guess,
    learning_rate=0.001,
    tolerance=1e-3,
    max_iterations=50,
)
optimised_brightness_2 = extended_results["optimised_params"][0]
optimised_scaling_2 = extended_results["optimised_params"][1]
optimised_edge_scaling_2 = extended_results["optimised_params"][2]
reconstructed_image_2 = reconstruct_image_extended(
    input_image=blurry_image,
    gaussian_variance=gaussian_variance,
    high_scaling_factor=optimised_scaling_2,
    brightness_factor=optimised_brightness_2,
    edge_scaling_factor=optimised_edge_scaling_2,
    overlap_percentage=overlap_percentage,
)

# Display results for Method 2
print("\nExtended Multi-Kernel Method Results:")
print(f"Optimised Brightness Factor: {optimised_brightness_2:.4f}")
print(f"Optimised High Scaling Factor: {optimised_scaling_2:.4f}")
print(f"Optimised Edge Scaling Factor: {optimised_edge_scaling_2:.4f}")
print(f"Minimum MSE: {extended_results['minimum_mse']:.6f}")
print(f"Convergence: {extended_results['converged']}")
print(f"Iterations: {extended_results['iterations']}")

# ----------- Plot Comparison -----------
plt.figure(figsize=(20, 5))

# Noisy (blurry) image
plt.subplot(1, 4, 1)
plt.imshow(blurry_image, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

# Reference sharp image
plt.subplot(1, 4, 2)
plt.imshow(sharp_reference, cmap="gray")
plt.title("Reference Sharp Image")
plt.axis("off")

# Reconstructed image using Method 1
plt.subplot(1, 4, 3)
plt.imshow(reconstructed_image_1, cmap="gray")
plt.title("Reconstructed: Two-Kernel")
plt.axis("off")

# Reconstructed image using Method 2
plt.subplot(1, 4, 4)
plt.imshow(reconstructed_image_2, cmap="gray")
plt.title("Reconstructed: Multi-Kernel")
plt.axis("off")

# Show plots
plt.tight_layout()
plt.show()