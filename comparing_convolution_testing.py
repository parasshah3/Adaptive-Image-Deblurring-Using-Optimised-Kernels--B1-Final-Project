# B1 Final Project: Adaptive Deblurring Using Optimised Kernels
# Paras Shah

import numpy as np
from scipy.signal import convolve2d
from skimage.metrics import mean_squared_error as mse
from adaptive_deblurring_functions import convolution_brute_force, convolution_AI, load_image

import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg', depending on your setup

def compare_mse(image_path, kernels):
    """
    Compare MSE of convolution methods (Brute-Force and FFT-Based).
    
    Args:
        image_path: File path to the test image.
        kernels: Dictionary of kernels (name: (kernel_array, description)).
    
    Returns:
        mse_results: Dictionary containing MSE for each kernel and method.
    """
    mse_results = {}
    image = load_image(image_path)
    print(f"Loaded Image: {image_path}, Shape: {image.shape}")

    for kernel_name, (kernel, kernel_description) in kernels.items():
        print(f"\nEvaluating Kernel: {kernel_name} ({kernel_description})")

        # Compute reference result using convolve2d
        reference_output = convolve2d(image, kernel, mode="valid")

        # Brute-Force method
        brute_force_output = convolution_brute_force(image, kernel)
        mse_brute_force = mse(reference_output, brute_force_output)

        # FFT-Based method
        fft_based_output = convolution_AI(image, kernel)
        mse_fft_based = mse(reference_output, fft_based_output)

        # Store results
        mse_results[kernel_name] = {
            "Kernel Description": kernel_description,
            "MSE Brute-Force": mse_brute_force,
            "MSE FFT-Based": mse_fft_based,
        }
        print(f"Brute-Force MSE: {mse_brute_force:.4f}, FFT-Based MSE: {mse_fft_based:.4f}")

    return mse_results

def display_mse_results(mse_results):
    """
    Display and visualize the MSE results for each kernel.

    Args:
        mse_results: Dictionary containing MSE for each kernel and method.

    Returns:
        None
    """
    kernel_names = []
    brute_force_mse = []
    fft_based_mse = []

    for kernel, results in mse_results.items():
        kernel_names.append(f"{kernel}: {results['Kernel Description']}")
        brute_force_mse.append(results["MSE Brute-Force"])
        fft_based_mse.append(results["MSE FFT-Based"])

    # Plot results
    x = np.arange(len(kernel_names))  # x-axis positions
    bar_width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar(x - bar_width / 2, brute_force_mse, width=bar_width, label="Brute-Force")
    plt.bar(x + bar_width / 2, fft_based_mse, width=bar_width, label="FFT-Based")

    # Add labels and title
    plt.xticks(x, kernel_names, rotation=45, ha="right")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.xlabel("Kernel")
    plt.title("MSE Comparison Between Brute-Force and FFT-Based Convolution Methods")
    plt.legend()
    plt.tight_layout()

    # Show plot
    plt.show()

if __name__ == "__main__":
    # File path to the test image
    image_path = "/Users/paras/Desktop/B1 Final Project/cameraman.tif"  # Replace with your test image path

    # Define kernels
    kernels = {
        "1": (np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]), "Sobel Edge Detection"),
        "2": (np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16, "Gaussian Blur (3x3)"),
        "3": (np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), "Laplacian Kernel"),
        "4": (np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]), "Emboss Kernel"),
        "5": (np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]), "Sharpen Kernel"),
        "6": (np.array([[1, 4, 6, 4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1, 4, 6, 4, 1]]) / 256, "Gaussian Blur (5x5)"),
        "7": (np.ones((7, 7)) / 49, "Uniform Box Blur (7x7)"),
        "8": (np.array([[0, -1, 0, -1, 0],
                        [-1, 2, -1, 2, -1],
                        [0, -1, 0, -1, 0],
                        [-1, 2, -1, 2, -1],
                        [0, -1, 0, -1, 0]]), "Pattern Detection Kernel"),
        "9": (np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]]), "Checkerboard Pattern Kernel (3x3)"),
        "10": (np.array([[0, -1, 0, -1, 0],
                         [-1, 2, -1, 2, -1],
                         [0, -1, 0, -1, 0],
                         [-1, 2, -1, 2, -1],
                         [0, -1, 0, -1, 0]]), "Pattern Kernel (5x5)")
    }

    # Run MSE comparison
    mse_results = compare_mse(image_path, kernels)

    # Display the results
    display_mse_results(mse_results)