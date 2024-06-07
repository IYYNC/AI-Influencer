# Instructions for Navigating

## Files and Descriptions:

1. **Final Project Report: Creating an AI Influencer**: The comprehensive report detailing the entire project, including objectives, methodologies, results, and conclusions.
   
2. **README.md**: The main project overview and setup instructions. Ensure you review this document first for an understanding of the project structure.

3. **getty_scraper.ipynb**: A Jupyter notebook used for scraping images from Getty Images. This file includes the code and instructions for collecting the dataset used in the project.

4. **image_generator.py**: This script generates images using the trained GAN models. It includes functions for loading the generator model and generating images based on random noise.

5. **preprocessing.py**: This script handles the preprocessing of images, including resizing and normalization. It also combines images from different categories into a single dataset and saves them as `.npy` files.

6. **train_gan_combined.py**: The script for training the third model (Combined Dataset GAN). It combines blonde female portraits and food images to generate diverse outputs.

7. **train_gan_portrait.py**: The script for training the first model (Standard GAN). It focuses on generating 128x128 pixel images of blonde female portraits.

8. **train_wgan_gp.py**: The script for training the second model (WGAN-GP). It uses the WGAN-GP approach to improve the realism and quality of the generated blonde female portraits.

## How to Use:

1. **Setting Up the Environment**:
   - Ensure you have Python and the necessary libraries installed. The primary libraries used include TensorFlow, NumPy, and PIL.
   - If using Google Colab, you can mount your Google Drive to access the files directly.

2. **Data Collection**:
   - Run `getty_scraper.ipynb` to scrape the necessary images from Getty Images. This notebook will save the images to your specified Google Drive folder.

3. **Preprocessing**:
   - Use `preprocessing.py` to preprocess the collected images. This script will resize, normalize, and combine the images, saving the preprocessed datasets as `.npy` files in your Google Drive.

4. **Model Training**:
   - Use `train_gan_portrait.py` to train the first model on blonde female portraits.
   - Use `train_wgan_gp.py` to train the second model using the WGAN-GP technique.
   - Use `train_gan_combined.py` to train the third model on the combined dataset of blonde portraits and food images.

5. **Generating Images**:
   - Run `image_generator.py` to generate new images using the trained GAN models. Ensure the appropriate model checkpoints are loaded from your Google Drive.

By following these instructions, you should be able to navigate through the project files, understand their purposes, and execute the necessary scripts to replicate the project results or further develop the models.
