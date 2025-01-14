import numpy as np
import tracemalloc
from scipy.signal import convolve2d
from adaptive_deblurring_functions import convolution_brute_force, convolution_AI, load_image

def measure_memory_usage(image, kernel, func, mode="valid"):
    """
    Measures the peak memory usage of a convolution function.

    Args:
        image: Input image as a numpy array.
        kernel: Convolution kernel as a numpy array.
        func: Convolution function to measure.
        mode: Convolution mode (only used for convolve2d).

    Returns:
        peak_memory: Peak memory usage in kilobytes.
    """
    tracemalloc.start()
    if func == convolve2d:
        func(image, kernel, mode=mode)  # Use the 'valid' mode for convolve2d
    else:
        func(image, kernel)  # For other methods
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak_memory / 1024  # Convert to KB

def memory_test(image_path, kernels):
    """
    Test peak memory usage for all kernels using three convolution methods.

    Args:
        image_path: File path to the image to be used for testing.
        kernels: Dictionary of kernels with names and descriptions.

    Returns:
        results: Dictionary of peak memory usage for each method and kernel.
    """
    image = load_image(image_path)
    results = {"Brute-Force": [], "FFT-Based": [], "convolve2d": []}

    for kernel_name, (kernel, kernel_description) in kernels.items():
        print(f"Evaluating Kernel: {kernel_name} ({kernel_description})")

        # Measure memory usage for Brute-Force method
        memory_brute = measure_memory_usage(image, kernel, convolution_brute_force)
        results["Brute-Force"].append(memory_brute)
        print(f"Brute-Force Peak Memory: {memory_brute:.2f} KB")

        # Measure memory usage for FFT-Based method
        memory_fft = measure_memory_usage(image, kernel, convolution_AI)
        results["FFT-Based"].append(memory_fft)
        print(f"FFT-Based Peak Memory: {memory_fft:.2f} KB")

        # Measure memory usage for convolve2d method
        memory_convolve2d = measure_memory_usage(image, kernel, convolve2d)
        results["convolve2d"].append(memory_convolve2d)
        print(f"convolve2d Peak Memory: {memory_convolve2d:.2f} KB")

    return results

def display_memory_results(results, kernels):
    """
    Plot the memory usage results for all convolution methods.

    Args:
        results: Dictionary of memory usage for each method.
        kernels: Dictionary of kernels.

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    kernel_names = [f"{key} ({desc})" for key, (_, desc) in kernels.items()]
    x = np.arange(len(kernels))  # X-axis positions
    bar_width = 0.25

    plt.figure(figsize=(14, 7))
    plt.bar(x - bar_width, results["Brute-Force"], width=bar_width, label="Brute-Force")
    plt.bar(x, results["FFT-Based"], width=bar_width, label="FFT-Based")
    plt.bar(x + bar_width, results["convolve2d"], width=bar_width, label="convolve2d")

    # Add labels and title
    plt.xticks(x, kernel_names, rotation=45, ha="right")
    plt.ylabel("Peak Memory Usage (KB)")
    plt.xlabel("Kernel")
    plt.title("Peak Memory Usage Across Kernels and Methods")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
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

    # Test image path (replace with the actual path to your image)
    image_path = "/Users/paras/Desktop/B1 Final Project/cameraman.tif"

    # Measure memory usage
    memory_results = memory_test(image_path, kernels)

    # Plot memory usage results
    display_memory_results(memory_results, kernels)