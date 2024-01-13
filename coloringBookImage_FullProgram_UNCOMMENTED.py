import cv2
import numpy as np
from PIL import Image, ImageEnhance
from skimage import morphology
import os
import tkinter as tk
from tkinter import filedialog, messagebox

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
def reduce_details(image):
    # Implement your detail reduction logic here
    return image

# Function to process a single image
def process_single_image(image_path):
    image = cv2.imread(image_path)  # Read the image using OpenCV
    # Apply image processing functions here
    image = upscale_image(image)
    image = adjust_brightness_contrast(Image.fromarray(image), 1.2, 1.2)  # Example brightness and contrast adjustment
    image = convert_to_grayscale(image)  # Example grayscale conversion
    image = simplify_shapes(image)  # Example shape simplification
    # Additional image processing steps can be added here
    return image  # Return the processed image

# Function to process multiple images in a folder
def process_folder(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, file_name)
            processed_image = process_single_image(file_path)  # Process each image
            # Save the processed image with your existing saving functionality
            # Example: cv2.imwrite('processed_' + file_name, processed_image)

# Tkinter GUI class for the application
class ColoringBookApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Coloring Book Image Processor")
        self.create_widgets()

    def run(self):
        self.root.mainloop()

    def create_widgets(self):
        open_button = tk.Button(self.root, text="Open File/Folder", command=self.process_image)
        open_button.pack()

    def process_image(self):
        path = self.select_image_or_folder()
        if path:
            self.handle_processing(path)
            messagebox.showinfo("Info", "Processing completed.")
        else:
            messagebox.showinfo("Error", "No file or folder selected.")

    def select_image_or_folder(self):
        choice = messagebox.askquestion("Choose", "Process a single file or a folder?", icon='question', type='yesno')
        return filedialog.askopenfilename() if choice == 'yes' else filedialog.askdirectory()

    def handle_processing(self, path):
        if os.path.isdir(path):
            process_folder(path)
        elif os.path.isfile(path):
            processed_image = process_single_image(path)
            # Save processed image
            # Implement saving logic here, e.g., self.save_image(Image.fromarray(processed_image), os.path.splitext(os.path.basename(path))[0])

    def save_image(self, image, default_name="processed_image"):
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                initialfile=default_name,
                                                filetypes=[("PNG files", "*.png"), ("All Files", "*.*")])
        if file_path:
            image.save(file_path)
            messagebox.showinfo("Info", "Image saved successfully.")
        else:
            messagebox.showinfo("Info", "Save operation cancelled.")

if __name__ == "__main__":
    app = ColoringBookApp()
    app.run()
