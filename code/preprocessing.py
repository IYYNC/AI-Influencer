# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
from google.colab import drive
from tensorflow.keras.preprocessing.image import img_to_array

# Mount Google Drive
drive.mount('/content/drive')

# Define paths to the folders in Google Drive
BLONDE_PHOTOS_PATH = '/content/drive/MyDrive/blonde_woman'
FOOD_PHOTOS_PATH = '/content/drive/MyDrive/food_photos/'
PROCESSED_IMAGES_PATH = '/content/drive/MyDrive/ProcessedImages/'

# Create directory for processed images if it doesn't exist
os.makedirs(PROCESSED_IMAGES_PATH, exist_ok=True)

def preprocess_image(image_path, target_size=(128, 128)):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = img_to_array(img)
        img_array = (img_array / 127.5) - 1  # Normalize to [-1, 1]
        return img_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def load_images_from_folder(folder_path, target_size=(128, 128)):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    print(f"Found {len(image_files)} files in folder {folder_path}")
    images = []
    for image_file in image_files:
        img_array = preprocess_image(image_file, target_size)
        if img_array is not None:
            images.append(img_array)
        else:
            print(f"Failed to preprocess image {image_file}")
    print(f"Loaded {len(images)} images from folder {folder_path}")
    return np.array(images)

def save_npy_to_drive(data, filename):
    save_path = os.path.join(PROCESSED_IMAGES_PATH, filename)
    np.save(save_path, data)
    print(f"Successfully saved {save_path} to Google Drive.")

def main():
    blonde_images = load_images_from_folder(BLONDE_PHOTOS_PATH)
    food_images = load_images_from_folder(FOOD_PHOTOS_PATH)

    # Combine the images
    combined_images = np.concatenate((blonde_images, food_images), axis=0)
    np.random.shuffle(combined_images)

    # Save preprocessed images
    save_npy_to_drive(blonde_images, 'blonde_images.npy')
    save_npy_to_drive(food_images, 'food_images.npy')
    save_npy_to_drive(combined_images, 'combined_images.npy')

if __name__ == "__main__":
    main()
