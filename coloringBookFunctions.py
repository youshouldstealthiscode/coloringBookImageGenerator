import cv2  # Imports the OpenCV library, which is used for image processing
import numpy as np  # Imports the NumPy library, which is used for numerical operations on arrays
from PIL import Image, ImageEnhance  # Imports the Image and ImageEnhance modules from PIL (Python Imaging Library) for image manipulation
from skimage import morphology  # Imports the morphology module from skimage (Scikit-image) for image morphology operations

# Function to check image resolution
def check_image_resolution(image_path):
    with Image.open(image_path) as img:  # Opens the image file from the given path
        if 'dpi' in img.info:  # Checks if the DPI (Dots Per Inch) information is available in the image metadata
            dpi = img.info['dpi']  # Retrieves the DPI value
            return dpi  # Returns the DPI value
        else:
            return (72, 72)  # Returns a default DPI of 72x72 if DPI information is not available in the image metadata

# Function to upscale image to 300dpi
def upscale_image(image, target_dpi=300):
    current_dpi = check_image_resolution(image)  # Calls the function to check the current DPI of the image
    scale_factor = target_dpi / min(current_dpi)  # Calculates the scaling factor to upscale the image to the target DPI
    new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))  # Calculates the new size of the image after scaling
    upscaled_img = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)  # Resizes the image to the new size using OpenCV's resize function
    return upscaled_img  # Returns the upscaled image

# Function to adjust brightness and contrast
def adjust_brightness_contrast(image_pil, brightness_factor, contrast_factor):
    enhancer = ImageEnhance.Brightness(image_pil)  # Creates a brightness enhancer for the image
    image_pil = enhancer.enhance(brightness_factor)  # Adjusts the brightness of the image
    enhancer = ImageEnhance.Contrast(image_pil)  # Creates a contrast enhancer for the image
    image_pil = enhancer.enhance(contrast_factor)  # Adjusts the contrast of the image
    return image_pil  # Returns the image with adjusted brightness and contrast

# Function to reduce noise
def reduce_noise(image):
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)  # Applies noise reduction using Non-Local Means Denoising
    return denoised_image  # Returns the denoised image

# Function to convert image to grayscale
def convert_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converts the image to grayscale using OpenCV's color conversion function
    return grayscale_image  # Returns the grayscale image

# Function to simplify shapes
def simplify_shapes(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Finds the contours in the image
    simplified_image = image.copy()  # Creates a copy of the original image to draw the simplified shapes
    for contour in contours:  # Iterates through each contour
        epsilon = 0.01 * cv2.arcLength(contour, True)  # Calculates epsilon for contour approximation
        approx = cv2.approxPolyDP(contour, epsilon, True)  # Approximates the shape of the contour
        cv2.drawContours(simplified_image, [approx], 0, (255, 255, 255), -1)  # Draws the approximated contour on the image
    return simplified_image  # Returns the image with simplified shapes

# Function to apply edge detection
def edge_detection(image):
    edges = cv2.Canny(image, 100, 200)  # Applies the Canny edge detection algorithm to the image
    return edges  # Returns the image with detected edges

# Function to reduce details
def reduce_details(image, area_threshold=100):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Finds the contours in the image
    for contour in contours:  # Iterates through each contour
        if cv2.contourArea(contour) < area_threshold:  # Checks if the contour area is smaller than the threshold
            cv2.drawContours(image, [contour], 0, (0, 0, 0), -1)  # Fills the small contours with black color
    return image  # Returns the image with reduced details

# Function to thin lines
def thin_lines(image):
    thinned_image = morphology.thin(image)  # Applies thinning operation to the image using skimage's morphology module
    return thinned_image  # Returns the image with thinned lines

# Placeholder functions for varying line widths, removing overlapping elements, and simplifying background
def vary_line_widths(image):
    # Custom logic to vary line widths
    return image

def remove_overlapping_elements(image):
    # Custom logic to remove overlaps
    return image

def simplify_background(image):
    # Custom logic to simplify background
    return image

# Function to sharpen lines
def sharpen_lines(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Defines a kernel for sharpening
    sharpened_image = cv2.filter2D(image, -1, kernel)  # Applies the sharpening filter to the image
    return sharpened_image  # Returns the image with sharpened lines

# Function to smooth line edges
def smooth_line_edges(image):
    smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)  # Applies Gaussian blur to smooth the edges of the lines
    return smoothed_image  # Returns the image with smoothed line edges

# Function to enhance contrast
def enhance_contrast(image_pil):
    enhanced_image = ImageEnhance.Contrast(image_pil).enhance(2.0)  # Enhances the contrast of the image
    return enhanced_image  # Returns the image with enhanced contrast

# Function to set export parameters and save final image
def set_export_parameters_save(image_pil, output_path):
    image_pil.save(output_path, format="PNG", dpi=(300, 300))  # Saves the image with specified parameters (format and DPI)

# Main function to process the image
def process_image(image_path, brightness_factor=1.0, contrast_factor=1.0):
    # Load image
    original_image = Image.open(image_path)  # Opens the image from the given path
    image = cv2.imread(image_path)  # Reads the image using OpenCV

    # Check and upscale resolution
    image = upscale_image(image)  # Calls the function to upscale the image resolution

    # Adjust brightness and contrast
    image_pil = adjust_brightness_contrast(original_image, brightness_factor, contrast_factor)  # Adjusts the brightness and contrast of the image
    image = np.array(image_pil)  # Converts the PIL image to a NumPy array

    # Reduce noise
    image = reduce_noise(image)  # Calls the function to reduce noise in the image

    # Convert to grayscale and simplify shapes
    grayscale_image = convert_to_grayscale(image)  # Converts the image to grayscale
    simplified_image = simplify_shapes(grayscale_image)  # Simplifies the shapes in the grayscale image

    # Edge detection and detail reduction
    edges = edge_detection(simplified_image)  # Detects edges in the simplified image
    reduced_detail_image = reduce_details(edges)  # Reduces details in the image with detected edges

    # Enhance line art
    thinned_image = thin_lines(reduced_detail_image)  # Thins the lines in the image
    varied_width_image = vary_line_widths(thinned_image)  # Varies the width of lines
    sharpened_image = sharpen_lines(varied_width_image)  # Sharpens the lines in the image
    smoothed_image = smooth_line_edges(sharpened_image)  # Smooths the edges of the lines

    # Further enhancements and saving
    final_image_pil = Image.fromarray(smoothed_image)  # Converts the processed image to a PIL Image
    final_contrasted_image = enhance_contrast(final_image_pil)  # Enhances the contrast of the final image
    set_export_parameters_save(final_contrasted_image, "final_output.png")  # Saves the final image with export parameters

# Example usage
process_image("path_to_image.jpg")  # Calls the process_image function with the path to the image
