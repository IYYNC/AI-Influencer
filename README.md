# AI-Influencer
## Project Overview
This project aims to create an AI influencer persona using Generative Adversarial Networks (GANs). The generated AI influencer will be used for social media content creation, specifically targeting platforms like Instagram. The project involves the following key steps:

1. Scraping and storing images.
2. Setting up the environment on Google Cloud Platform (GCP).
3. Training two different models.
4. Evaluating and improving the models.
5. Utilizing existing AI tools.
6. Planning for social media strategy.
## Step 1: Scraping and Storing Images
We scraped images from Getty Images, focusing on two categories:

- 5000 images of blonde female portraits.
- 5000 images of various food items.
These images were stored in Google Drive for easy access and management.

## Step 2: Setting Up GCP Environment
We set up a Virtual Machine (VM) on GCP to train our models, leveraging the computational power of the cloud.

## Step 3: Training the Models
### Model 1: Training on Blonde Female Portraits
**Objective:** Generate high-quality images of blonde female portraits.

**Methodology:**

- Load blonde female images from Google Drive.
- Preprocess the images (resize, normalize).
- Build and train a GAN using the following architecture:
  - Generator: Dense, Conv2DTranspose, LeakyReLU, and BatchNormalization layers.
  - Discriminator: Conv2D, LeakyReLU, Dropout, and Dense layers.
- Save the trained model locally.
### Model 2: Training on Combined Blonde Female Portraits and Food Images
**Objective:** Generate images combining blonde female portraits and food items.

**Methodology:**

- Load blonde female images and food images from Google Drive.
- Preprocess the images (resize, normalize).
- Combine the two categories of images into a single dataset.
- Build and train a GAN using the same architecture as Model 1.
- Save the trained model locally.
**Dealing with Combining and Labeling Two Different Categories of Photos:**

- **Combining the Datasets:** The blonde female images and food images were loaded separately and then combined into a single dataset. This ensures that the GAN is trained on a diverse set of images that include both categories.
- **Preprocessing:** Each image, regardless of category, was resized to a uniform size of 128x128 pixels and normalized to a range of [-1, 1]. This standardization ensures consistency in the input data for the GAN.
- **Training Approach:** The combined dataset was shuffled to ensure that the GAN receives a mix of both categories during each training iteration. This helps the generator learn to produce images that might creatively combine elements of both categories.
## Step 4: Evaluating and Improving the Models
Both models were trained and evaluated for their ability to generate high-quality images. Despite the training efforts, the models initially produced low-quality images. Various techniques were explored to improve the output quality, such as adjusting the network architecture and tuning hyperparameters.

## Step 5: Utilizing Existing AI Tools
Recognizing the challenges in generating high-quality images, we explored using existing AI tools like Leonardo AI to generate content. These tools offered better quality and stability, which were crucial for our project goals.

## Step 6: Planning for Social Media Strategy
With the generated images, we plan to build an Instagram account for our AI influencer persona. Our strategy includes:

- Regularly posting generated content.
- Engaging with followers.
- Monitoring metrics like follower count and engagement rates.
- Deciding on further investments (e.g., subscriptions for AI tools) based on the success of the account.
## Conclusion
This project demonstrates the potential of GANs in creating AI-generated content for social media. While there are challenges in achieving high-quality outputs, leveraging existing AI tools and focusing on a strategic social media plan can lead to successful implementation and monetization of an AI influencer persona.
## Credits
This project was a collaborative effort, and I would like to thank our team members for their contributions:

- **Jodie Chen**
- **Florentina Santosa**
- **Manickashree Thayumana Sundaram**
