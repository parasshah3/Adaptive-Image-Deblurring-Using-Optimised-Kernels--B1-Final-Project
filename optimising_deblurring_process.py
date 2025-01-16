import numpy as np
import matplotlib.pyplot as plt
from adaptive_deblurring_functions import load_image, optimise_parameters

# Paths to the input (blurry) and reference (sharp) images
blurry_image_path = "/Users/paras/Desktop/B1 Final Project/defocused building.JPG"
sharp_image_path = "/Users/paras/Desktop/B1 Final Project/sharp_building.JPG"

# Load the blurry and sharp reference images
blurry_image = load_image(blurry_image_path)
sharp_reference = load_image(sharp_image_path)

# Parameters for Gaussian variance and overlap percentage
gaussian_variance = 1.0  # Adjust as needed
overlap_percentage = 90  # Overlap percentage for patches

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

# Visualise the reconstructed image with optimised parameters
from adaptive_deblurring_functions import reconstruct_image

optimised_brightness = optimisation_results['optimised_params'][0]
optimised_scaling = optimisation_results['optimised_params'][1]

reconstructed_image = reconstruct_image(
    input_image=blurry_image,
    gaussian_variance=gaussian_variance,
    high_scaling_factor=optimised_scaling,
    brightness_factor=optimised_brightness,
    overlap_percentage=overlap_percentage,
)

# Plot the original blurry image, reference image, and optimised reconstructed image
plt.figure(figsize=(15, 5))

# Original blurry image
plt.subplot(1, 3, 1)
plt.imshow(blurry_image, cmap="gray")
plt.title("Blurry Image")
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