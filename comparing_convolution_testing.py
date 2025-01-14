# B1 Final Project: Adaptive Deblurring Using Optimised Kernels
# Paras Shah

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from adaptive_deblurring_functions import convolution_brute_force, convolution_AI, load_image

def time_convolution_methods(image_paths, kernels):
    """
    Time the execution of three convolution methods for each image and kernel.
    
    Args:
        image_paths: List of file paths to the images.
        kernels: Dictionary of kernels (name: (kernel_array, description)).
    
    Returns:
        results: Dictionary with average time taken for each kernel by each method.
    """
    results = {method: [] for method in ["Brute-Force", "FFT-Based", "convolve2d"]}
    
    for kernel_name, (kernel, kernel_description) in kernels.items():
        print(f"\nTiming convolution methods for Kernel: {kernel_name} ({kernel_description})")
        times = {"Brute-Force": [], "FFT-Based": [], "convolve2d": []}

        for path in image_paths:
            image = load_image(path)
            print(f"Processing Image: {path}, Shape: {image.shape}")

            # Time Brute-Force method
            start_time = time.time()
            convolution_brute_force(image, kernel)
            times["Brute-Force"].append(time.time() - start_time)

            # Time FFT-Based method
            start_time = time.time()
            convolution_AI(image, kernel)
            times["FFT-Based"].append(time.time() - start_time)

            # Time convolve2d method
            start_time = time.time()
            convolve2d(image, kernel, mode="valid")
            times["convolve2d"].append(time.time() - start_time)

        # Calculate average times for the current kernel
        for method in times:
            avg_time_ms = np.mean(times[method]) * 1000  # Convert to milliseconds
            results[method].append(avg_time_ms)
            print(f"Average time for {method}: {avg_time_ms:.4f} ms")

    return results

def plot_efficiency_results(kernels, results):
    """
    Plot the average execution times for each method and kernel.

    Args:
        kernels: Dictionary of kernels (name: (kernel_array, description)).
        results: Dictionary with average times for each method.

    Returns:
        None
    """
    kernel_names = [f"{key}: {desc}" for key, (_, desc) in kernels.items()]
    x = np.arange(len(kernels))  # x-axis positions

    # Bar widths
    bar_width = 0.2

    # Create a grouped bar chart with log scale
    plt.figure(figsize=(14, 7))
    plt.bar(x - bar_width, results["Brute-Force"], width=bar_width, label="Brute-Force")
    plt.bar(x, results["FFT-Based"], width=bar_width, label="FFT-Based")
    plt.bar(x + bar_width, results["convolve2d"], width=bar_width, label="convolve2d")

    # Add labels and title
    plt.xticks(x, kernel_names, rotation=45, ha="right")
    plt.ylabel("Average Time (milliseconds, log scale)")
    plt.xlabel("Kernel (Type: Description)")
    plt.title("Average Convolution Times Across Kernels and Methods")
    plt.yscale("log")  # Logarithmic scale
    plt.legend()
    plt.tight_layout()

    # Show plot
    plt.show()

if __name__ == "__main__":
    # File paths for the 5 test images
    image_paths = [
        "/Users/paras/Desktop/B1 Final Project/cameraman.tif",  # Replace with actual paths
        "/Users/paras/Desktop/B1 Final Project/lena_color_512.tif",
        "/Users/paras/Desktop/B1 Final Project/mandril_color.tif",
        "/Users/paras/Desktop/B1 Final Project/MRI Brain Scan 1.jpg",
        "/Users/paras/Desktop/B1 Final Project/pirate.tif",]

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

    # Run timing tests
    results = time_convolution_methods(image_paths, kernels)

    # Plot the results
    plot_efficiency_results(kernels, results)