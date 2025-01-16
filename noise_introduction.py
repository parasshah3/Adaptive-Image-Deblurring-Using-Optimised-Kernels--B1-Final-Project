import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from adaptive_deblurring_functions import add_noise

# Example Usage
if __name__ == "__main__":
    # Load a grayscale image
    image_path = "/Users/paras/Desktop/B1 Final Project/sharp_building.JPG"
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_array = np.array(img)

    # Add Gaussian noise
    gaussian_noisy_img = add_noise(img_array, noise_type="gaussian", mean=0, var=0.1)

    # Add Salt-and-Pepper noise
    sp_noisy_img = add_noise(img_array, noise_type="salt_and_pepper", salt_prob=0.05, pepper_prob=0.05)

    # Display the images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img_array, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gaussian_noisy_img, cmap="gray")
    plt.title("Gaussian Noise")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(sp_noisy_img, cmap="gray")
    plt.title("Salt-and-Pepper Noise")
    plt.axis("off")

    plt.tight_layout()
    plt.show()