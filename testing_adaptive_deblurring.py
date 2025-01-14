# Testing script for get_global_properties function

from adaptive_deblurring_functions import get_global_properties, load_image

# Paths to the test images
image_paths = [
    "/Users/paras/Desktop/B1 Final Project/cameraman.tif",  # Replace with actual paths
    "/Users/paras/Desktop/B1 Final Project/lena_color_512.tif",
    "/Users/paras/Desktop/B1 Final Project/mandril_color.tif",
    "/Users/paras/Desktop/B1 Final Project/MRI Brain Scan 1.jpg",
    "/Users/paras/Desktop/B1 Final Project/pirate.tif",
]

# Analyze each image
for image_path in image_paths:
    print(f"Analyzing Image: {image_path}")
    # Load the image as a grayscale numpy array
    image = load_image(image_path, grayscale=True)
    
    # Compute global properties
    resolution, global_variance, global_gradient_magnitude = get_global_properties(image)
    
    # Output the results
    print(f"Global Variance (Type: {type(global_variance)}): {global_variance}")
    print(f"Resolution: {resolution}")
    print(f"Global Variance (σ²_global): {global_variance:.4f}")
    print(f"Global Gradient Magnitude (G_global): {global_gradient_magnitude:.4f}")
    print("-" * 40)