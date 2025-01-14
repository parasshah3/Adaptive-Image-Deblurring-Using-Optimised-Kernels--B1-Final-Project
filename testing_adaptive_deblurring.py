from adaptive_deblurring_functions import load_image, divide_into_patches

# Paths to the test images
image_paths = [
    "/Users/paras/Desktop/B1 Final Project/cameraman.tif",  # Replace with actual paths
    "/Users/paras/Desktop/B1 Final Project/lena_color_512.tif",
    "/Users/paras/Desktop/B1 Final Project/mandril_color.tif",
    "/Users/paras/Desktop/B1 Final Project/MRI Brain Scan 1.jpg",
    "/Users/paras/Desktop/B1 Final Project/pirate.tif",
]

# Test the function for each image
for image_path in image_paths:
    print(f"Analyzing Image: {image_path}")
    
    # Load the image as a grayscale numpy array
    image = load_image(image_path, grayscale=True)
    
    # Divide the image into patches
    patches = divide_into_patches(image, overlap_percentage=50)
    
    # Output the results
    print(f"Image Shape: {image.shape}")
    print(f"Number of Patches: {len(patches)}")
    print(f"Patch Shape: {patches[0].shape if patches else 'N/A'}")
    print("-" * 40)