import numpy as np
import matplotlib.pyplot as plt
from adaptive_deblurring_functions_AI import (
    load_image,
    add_noise,
    reconstruct_image_extended,
    gradient_descent_optimisation
)

# Path to the sharp (reference) image
sharp_image_path = "/Users/paras/Desktop/B1 Final Project/pirate.tif"  # Update with the correct path

# Load the sharp reference image
sharp_reference = load_image(sharp_image_path)

# Simulate a noisy input image
noise_type = "poisson"  # Options: "gaussian", "salt_and_pepper", "poisson", "speckle"
mean = 0  # Mean for Gaussian noise
variance = 0.01  # Variance for Gaussian noise
blurry_image = add_noise(sharp_reference, noise_type, mean, variance)

# Parameters for Gaussian variance and overlap percentage
gaussian_variance = 1.0  # Variance for Gaussian blur kernel
overlap_percentage = 75  # Overlap percentage for patches

# Initial guesses for the optimisation parameters
initial_guess = [1.01, 1.11, 0.65]  # [brightness_factor, high_scaling_factor, edge_scaling_factor]

# Perform optimisation using gradient descent
optimisation_results = gradient_descent_optimisation(
    input_image=blurry_image,
    reference_image=sharp_reference,
    gaussian_variance=gaussian_variance,
    overlap_percentage=overlap_percentage,
    initial_guess=initial_guess,
    learning_rate=0.001,
    tolerance=1e-3,
    max_iterations=50
)

# Display optimisation results
print("Optimisation Results:")
print(f"Optimised Brightness Factor: {optimisation_results['optimised_params'][0]:.4f}")
print(f"Optimised High Scaling Factor: {optimisation_results['optimised_params'][1]:.4f}")
print(f"Optimised Edge Scaling Factor: {optimisation_results['optimised_params'][2]:.4f}")
print(f"Minimum MSE: {optimisation_results['minimum_mse']:.6f}")
print(f"Convergence: {optimisation_results['converged']}")
print(f"Iterations: {optimisation_results['iterations']}")

# Reconstruct the image using optimised parameters
optimised_brightness = optimisation_results['optimised_params'][0]
optimised_scaling = optimisation_results['optimised_params'][1]
optimised_edge_scaling = optimisation_results['optimised_params'][2]

reconstructed_image = reconstruct_image_extended(
    input_image=blurry_image,
    gaussian_variance=gaussian_variance,
    high_scaling_factor=optimised_scaling,
    brightness_factor=optimised_brightness,
    edge_scaling_factor=optimised_edge_scaling,
    overlap_percentage=overlap_percentage
)

# Plot the noisy image, reference image, and reconstructed image
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