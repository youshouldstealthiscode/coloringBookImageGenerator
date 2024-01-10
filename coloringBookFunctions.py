import cv2
import numpy as np
from PIL import Image, ImageEnhance
from skimage import morphology

# Function to check image resolution
def check_image_resolution(image_path):
    with Image.open(image_path) as img:
        if 'dpi' in img.info:
            dpi = img.info['dpi']
            return dpi
        else:
            return (72, 72)  # Default DPI if not specified

# Function to upscale image to 300dpi
def upscale_image(image, target_dpi=300):
    current_dpi = check_image_resolution(image)
    scale_factor = target_dpi / min(current_dpi)
    new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    upscaled_img = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return upscaled_img

# Function to adjust brightness and contrast
def adjust_brightness_contrast(image_pil, brightness_factor, contrast_factor):
    enhancer = ImageEnhance.Brightness(image_pil)
    image_pil = enhancer.enhance(brightness_factor)
    enhancer = ImageEnhance.Contrast(image_pil)
    image_pil = enhancer.enhance(contrast_factor)
    return image_pil

# Function to reduce noise
def reduce_noise(image):
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised_image

# Function to convert image to grayscale
def convert_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

# Function to simplify shapes
def simplify_shapes(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified_image = image.copy()
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(simplified_image, [approx], 0, (255, 255, 255), -1)
    return simplified_image

# Function to apply edge detection
def edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

# Function to reduce details
def reduce_details(image, area_threshold=100):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < area_threshold:
            cv2.drawContours(image, [contour], 0, (0, 0, 0), -1)
    return image

# Function to thin lines
def thin_lines(image):
    thinned_image = morphology.thin(image)
    return thinned_image

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
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# Function to smooth line edges
def smooth_line_edges(image):
    smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
    return smoothed_image

# Function to enhance contrast
def enhance_contrast(image_pil):
    enhanced_image = ImageEnhance.Contrast(image_pil).enhance(2.0)
    return enhanced_image

# Function to set export parameters and save final image
def set_export_parameters_save(image_pil, output_path):
    image_pil.save(output_path, format="PNG", dpi=(300, 300))

# Main function to process the image
def process_image(image_path, brightness_factor=1.0, contrast_factor=1.0):
    # Load image
    original_image = Image.open(image_path)
    image = cv2.imread(image_path)

    # Check and upscale resolution
    image = upscale_image(image)

    # Adjust brightness and contrast
    image_pil = adjust_brightness_contrast(original_image, brightness_factor, contrast_factor)
    image = np.array(image_pil)

    # Reduce noise
    image = reduce_noise(image)

    # Convert to grayscale and simplify shapes
    grayscale_image = convert_to_grayscale(image)
    simplified_image = simplify_shapes(grayscale_image)

    # Edge detection and detail reduction
    edges = edge_detection(simplified_image)
    reduced_detail_image = reduce_details(edges)

    # Enhance line art
    thinned_image = thin_lines(reduced_detail_image)
    varied_width_image = vary_line_widths(thinned_image)
    sharpened_image = sharpen_lines(varied_width_image)
    smoothed_image = smooth_line_edges(sharpened_image)

    # Further enhancements and saving
    final_image_pil = Image.fromarray(smoothed_image)
    final_contrasted_image = enhance_contrast(final_image_pil)
    set_export_parameters_save(final_contrasted_image, "final_output.png")

# Example usage
process_image("path_to_image.jpg")
