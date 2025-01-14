# Testing script for kernel and patch size selection

from adaptive_deblurring_functions import load_image, get_kernel_patch_sizes

# Paths to the test images
image_paths = [
    "/Users/paras/Desktop/B1 Final Project/cameraman.tif",
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
    
    # Get kernel and patch sizes
    properties = get_kernel_patch_sizes(image)
    
    # Output the results
    print(f"Resolution: {properties['resolution']}")
    print(f"Global Variance (σ²_global): {properties['global_variance']:.4f}")
    print(f"Kernel Size: {properties['kernel_size']}")
    print(f"Patch Size: {properties['patch_size']}")
    print("-" * 40)