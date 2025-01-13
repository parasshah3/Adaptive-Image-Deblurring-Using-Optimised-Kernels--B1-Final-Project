# Comparing convolution functions
# Paras Shah - B1 Final Project: Adaptive Deblurring Using Optimised Kernels

import numpy as np
from scipy.signal import convolve2d
from adaptive_deblurring_functions import convolution_brute_force, convolution_AI  # Import both functions

def test_convolution_methods():
    """
    Test the convolution_brute_force and convolution_AI functions with various test cases
    using convolve2d for verification.
    """
    print("\n===== Running Tests for Brute-Force and FFT-Based Convolution with convolve2d Verification =====\n")

    # Test Case 1: Simple 5x5 input and 3x3 kernel
    input_image_1 = np.array([[1, 2, 3, 4, 5],
                              [6, 7, 8, 9, 10],
                              [11, 12, 13, 14, 15],
                              [16, 17, 18, 19, 20],
                              [21, 22, 23, 24, 25]])
    kernel_1 = np.array([[1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1]])
    # Expected output using convolve2d
    expected_output_1 = convolve2d(input_image_1, kernel_1, mode='valid')
    # Outputs from both methods
    output_brute_1 = convolution_brute_force(input_image_1, kernel_1)
    output_ai_1 = convolution_AI(input_image_1, kernel_1)
    print("Test Case 1: Simple 5x5 input and 3x3 kernel")
    print("Expected Output:\n", expected_output_1)
    print("Brute-Force Output:\n", output_brute_1)
    print("FFT-Based Output:\n", output_ai_1)
    print("Brute-Force Pass:", np.allclose(output_brute_1, expected_output_1))
    print("FFT-Based Pass:", np.allclose(output_ai_1, expected_output_1))

    # Test Case 2: Gaussian-like kernel (3x3)
    input_image_2 = np.array([[10, 20, 30, 40],
                              [50, 60, 70, 80],
                              [90, 100, 110, 120],
                              [130, 140, 150, 160]])
    kernel_2 = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]])
    expected_output_2 = convolve2d(input_image_2, kernel_2, mode='valid')
    output_brute_2 = convolution_brute_force(input_image_2, kernel_2)
    output_ai_2 = convolution_AI(input_image_2, kernel_2)
    print("\nTest Case 2: Gaussian-like kernel")
    print("Expected Output:\n", expected_output_2)
    print("Brute-Force Output:\n", output_brute_2)
    print("FFT-Based Output:\n", output_ai_2)
    print("Brute-Force Pass:", np.allclose(output_brute_2, expected_output_2))
    print("FFT-Based Pass:", np.allclose(output_ai_2, expected_output_2))

    # Test Case 3: Edge case with all zeros (7x7 input and 3x3 kernel)
    input_image_3 = np.zeros((7, 7))
    kernel_3 = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
    expected_output_3 = convolve2d(input_image_3, kernel_3, mode='valid')
    output_brute_3 = convolution_brute_force(input_image_3, kernel_3)
    output_ai_3 = convolution_AI(input_image_3, kernel_3)
    print("\nTest Case 3: Input image with all zeros")
    print("Expected Output:\n", expected_output_3)
    print("Brute-Force Output:\n", output_brute_3)
    print("FFT-Based Output:\n", output_ai_3)
    print("Brute-Force Pass:", np.allclose(output_brute_3, expected_output_3))
    print("FFT-Based Pass:", np.allclose(output_ai_3, expected_output_3))

    # Test Case 4: Random input image and kernel (odd sizes only)
    np.random.seed(42)  # Set seed for reproducibility
    input_image_4 = np.random.randint(0, 10, (9, 9))
    kernel_4 = np.random.randint(-5, 5, (5, 5))
    expected_output_4 = convolve2d(input_image_4, kernel_4, mode='valid')
    output_brute_4 = convolution_brute_force(input_image_4, kernel_4)
    output_ai_4 = convolution_AI(input_image_4, kernel_4)
    print("\nTest Case 4: Random input image and 5x5 kernel")
    print("Input Image:\n", input_image_4)
    print("Kernel:\n", kernel_4)
    print("Expected Output:\n", expected_output_4)
    print("Brute-Force Output:\n", output_brute_4)
    print("FFT-Based Output:\n", output_ai_4)
    print("Brute-Force Pass:", np.allclose(output_brute_4, expected_output_4))
    print("FFT-Based Pass:", np.allclose(output_ai_4, expected_output_4))

if __name__ == "__main__":
    test_convolution_methods()