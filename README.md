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

## Step 2: Setting Up Colab Environment
We used Google Colab with a T4 GPU runtime to train our models. Colab provides a powerful and flexible environment for running intensive deep learning tasks with the added advantage of free access to high-performance GPUs.

## Step 3: Training the Models
### Model 1: Training on Blonde Female Portraits
**Objective:** Generate high-quality images of blonde female portraits.

**Methodology:**

- Load blonde female images from Google Drive.
- Preprocess the images (resize, normalize).
- Build and train a GAN using the following architecture:
  - Generator:
    - Dense layer to create a base structure.
    - LeakyReLU activation.
    - BatchNormalization for stability.
    - Multiple Conv2DTranspose layers to upscale the image.
    - Final Conv2D layer with 'tanh' activation to generate the image.

  - Discriminator:
    - Conv2D layers to downscale the image.
    - LeakyReLU activation.
    - Dropout layers to prevent overfitting.
    - Flatten layer followed by a Dense layer with 'sigmoid' activation to classify images as real or fake.
  - Training Process:
    - Train the discriminator on real and fake images.
    - Train the generator through the GAN model.
    - Save the trained model checkpoints at regular intervals.
### Model 2: Training on Blonde Female Portraits with WGAN-GP
**Objective:** Generate high-quality images of blonde female portraits using WGAN-GP to address mode collapse and improve training stability.

**Methodology:**

- Load blonde female images from Google Drive.
- Preprocess the images (resize, normalize).
- Build and train a WGAN-GP using the following architecture:
- Generator Architecture:

  - Dense layer to create a base structure.
  - LeakyReLU activation.
  - BatchNormalization for stability.
  - Multiple Conv2DTranspose layers to upscale the image.
  - Final Conv2D layer with 'tanh' activation to generate the image.
- Discriminator Architecture:

  - Conv2D layers to downscale the image.
  - LeakyReLU activation.
  - Dropout layers to prevent overfitting.
  - Flatten layer followed by a Dense layer.
- WGAN-GP Specifics:

  - Gradient penalty to enforce Lipschitz constraint.
  - Custom loss functions for generator and discriminator.
  - Adam optimizer with a lower learning rate for stable training.
- Training Process:
  
  - Implement gradient penalty to enforce Lipschitz constraint.
  - Define custom loss functions for generator and discriminator.
  - Use Adam optimizer with a lower learning rate for stable training.
  - Save the trained model checkpoints at regular intervals.
 
### Model 3: Training on Combined Blonde Female Portraits and Food Images
**Objective** Generate images combining blonde female portraits and food items.

**Methodology**

- Load blonde female images and food images from Google Drive.
- Preprocess the images (resize, normalize).
- Combine the two categories of images into a single dataset.
- Build and train a GAN using the same architecture as Model 1.
**Combining and Labeling Two Different Categories of Photos**

- Combining the Datasets: The blonde female images and food images were loaded separately and then combined into a single dataset to ensure diverse training data.
- Preprocessing: Each image, regardless of category, was resized to a uniform size of 128x128 pixels and normalized to a range of [-1, 1].
- Training Approach: The combined dataset was shuffled to ensure that the GAN receives a mix of both categories during each training iteration.
**Generator and Discriminator Architectures**

- The generator and discriminator for this model have similar architectures to Model 1 but are optimized for a diverse dataset.
**Training Process**

- Train the discriminator on real and fake images.
- Train the generator through the GAN model.
- Save the trained model checkpoints at regular intervals.

## Step 4: Evaluating and Improving the Models
Models were trained and evaluated for their ability to generate high-quality images. Despite the training efforts, the models initially produced low-quality images. Various techniques were explored to improve the output quality, such as adjusting the network architecture and tuning hyperparameters. WGAN-GP was implemented to address mode collapse and improve image diversity.

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
