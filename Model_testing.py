import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU
from PIL import Image, ImageEnhance
import cv2

# Define a custom Conv2DTranspose layer without 'groups' argument
class CustomConv2DTranspose(keras.layers.Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove 'groups' from kwargs if present
        super().__init__(*args, **kwargs)

# Load the generator model with custom objects
model_path = r"C:\Users\varsh\Downloads\generator_epoch_100.h5"
generator = tf.keras.models.load_model(
    model_path,
    custom_objects={'LeakyReLU': LeakyReLU, 'Conv2DTranspose': CustomConv2DTranspose}
)

# Function to convert an image to a sketch using OpenCV
def convert_to_sketch(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(invert, (111, 111), 0)
    inverted_blur = cv2.bitwise_not(blur)
    sketch = cv2.divide(gray, inverted_blur, scale=256)
    return sketch

# Function to load and preprocess the input sketch for the model
def load_and_preprocess_sketch(sketch, target_size=(1280, 1280)):
    img = Image.fromarray(sketch).convert("RGB")

    # Maintain aspect ratio during resizing
    original_size = img.size
    original_ratio = original_size[0] / original_size[1]
    if original_ratio > 1:  # Landscape
        new_width = target_size[0]
        new_height = int(target_size[0] / original_ratio)
    else:  # Portrait
        new_height = target_size[1]
        new_width = int(target_size[1] * original_ratio)

    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Create a new square canvas with white background and paste the resized image
    new_img = Image.new("RGB", target_size, (255, 255, 255))
    new_img.paste(img, ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2))

    # Normalize image to [-1, 1] for the model
    img_array = np.array(new_img) / 127.5 - 1
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    return img_tensor

# Function to upscale and brighten the generated output
def upscale_and_brighten_output(prediction, scale_factor=10, brightness_factor=1.5):
    output_image = np.clip((prediction[0].numpy() + 1) * 127.5, 0, 255).astype(np.uint8)
    output_img = Image.fromarray(output_image)

    # Upscale the image
    new_size = (output_img.width * scale_factor, output_img.height * scale_factor)
    output_img = output_img.resize(new_size, Image.LANCZOS)

    # Brighten the image
    enhancer = ImageEnhance.Brightness(output_img)
    output_img = enhancer.enhance(brightness_factor)
    return output_img

# Visualization function# Visualization function
def visualize_prediction(input_image_path, brightened_output):
    # Load the input image in true color
    input_image = Image.open(input_image_path).convert("RGB")

    plt.figure(figsize=(15, 7))

    # Display the actual input image in true color
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(np.array(input_image))  # Convert PIL image to NumPy array for correct display
    plt.axis('off')

    # Display the final output image
    plt.subplot(1, 2, 2)
    plt.title("Brightened and Upscaled Output")
    plt.imshow(np.array(brightened_output))  # Ensure the output image is displayed correctly
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Specify the path to your input image
input_image_path = r"C:\Users\varsh\Desktop\dataset\o_comp\12.jpg" # Replace with your image path

# Process the image
sketch_image = convert_to_sketch(input_image_path)
input_sketch = load_and_preprocess_sketch(sketch_image)
input_sketch = tf.expand_dims(input_sketch, axis=0)  # Add batch dimension

# Generate the output
predictions = generator(input_sketch, training=False)
brightened_output = upscale_and_brighten_output(predictions)

# Visualize input and output
visualize_prediction(input_image_path, brightened_output)
