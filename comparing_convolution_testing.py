# B1 Final Project: Adaptive Deblurring Using Optimised Kernels
# Paras Shah

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from adaptive_deblurring_functions import convolution_brute_force, convolution_AI, load_image


def visualize_convolution_results(original, brute_force, fft_result, convolve2d_result, kernel_description):
    """
    Visualize the original image and convolution results using matplotlib.
    Args:
        original: Original image as a 2D numpy array.
        brute_force: Convolution result from the brute-force method.
        fft_result: Convolution result from the FFT-based method.
        convolve2d_result: Convolution result from scipy's convolve2d.
        kernel_description: Description of the kernel filter being used.

    Returns:
        None
    """
    plt.figure(figsize=(16, 8))
    plt.suptitle(f"Convolution Results Using {kernel_description}", fontsize=16, fontweight="bold")

    # Original image
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    # Brute-force result
    plt.subplot(1, 4, 2)
    plt.title("Brute-Force Convolution")
    plt.imshow(brute_force, cmap="gray")
    plt.axis("off")

    # FFT-based result
    plt.subplot(1, 4, 3)
    plt.title("FFT-Based Convolution")
    plt.imshow(fft_result, cmap="gray")
    plt.axis("off")

    # convolve2d result
    plt.subplot(1, 4, 4)
    plt.title("convolve2d Convolution")
    plt.imshow(convolve2d_result, cmap="gray")
    plt.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    plt.show()


def test_convolution_with_image(filepath, kernel, kernel_description):
    """
    Test and compare convolution methods with a real image.
    Args:
        filepath: Path to the image file.
        kernel: 2D numpy array representing the kernel.
        kernel_description: Description of the kernel filter.

    Returns:
        None
    """
    # Load the image
    image = load_image(filepath)
    print(f"Loaded Image Shape: {image.shape}")

    # Apply brute-force convolution
    brute_force_result = convolution_brute_force(image, kernel)

    # Apply FFT-based convolution
    fft_result = convolution_AI(image, kernel)

    # Apply scipy's convolve2d for comparison
    convolve2d_result = convolve2d(image, kernel, mode="valid")

    # Visualize results
    visualize_convolution_results(image, brute_force_result, fft_result, convolve2d_result, kernel_description)


if __name__ == "__main__":
    # Path to the MRI brain scan image
    filepath = "/Users/paras/Desktop/B1 Final Project/MRI Brain Scan 1.jpg"

    # Define a test kernel (e.g., Sobel edge detection kernel)
    kernel = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])  # Sobel kernel for edge detection

    # Kernel description
    kernel_description = "Sobel Edge Detection Kernel"

    # Test convolution methods with the image
    test_convolution_with_image(filepath, kernel, kernel_description)