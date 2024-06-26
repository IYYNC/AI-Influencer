# -*- coding: utf-8 -*-

import os
import numpy as np
from google.colab import drive
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, LeakyReLU, Conv2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Mount Google Drive
drive.mount('/content/drive')

# Define paths
PROCESSED_IMAGES_PATH = '/content/drive/MyDrive/ProcessedImages/'
BLONDE_IMAGES_FILE = os.path.join(PROCESSED_IMAGES_PATH, 'blonde_images.npy')

# Load preprocessed images
blonde_images = np.load(BLONDE_IMAGES_FILE)
print(f"Loaded {blonde_images.shape[0]} images for training.")

# Define GAN components
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(256 * 8 * 8, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Reshape((8, 8, 256)))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(3, (7,7), activation='tanh', padding='same'))
    return model

def build_discriminator(input_shape=(128, 128, 3)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

def train_gan(gan, generator, discriminator, dataset, latent_dim, n_epochs=100, n_batch=128, save_interval=10):
    half_batch = int(n_batch / 2)
    for epoch in range(n_epochs):
        print(f"Starting epoch {epoch+1}/{n_epochs}")
        for i in range(int(dataset.shape[0] / n_batch)):
            print(f"Training epoch {epoch+1}/{n_epochs}, batch {i+1}/{int(dataset.shape[0] / n_batch)}")

            # Train Discriminator
            idx = np.random.randint(0, dataset.shape[0], half_batch)
            real_images = dataset[idx]
            real_labels = np.ones((half_batch, 1))

            noise = np.random.normal(0, 1, (half_batch, latent_dim))
            fake_images = generator.predict(noise)
            fake_labels = np.zeros((half_batch, 1))

            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            noise = np.random.normal(0, 1, (n_batch, latent_dim))
            valid_labels = np.ones((n_batch, 1))
            g_loss = gan.train_on_batch(noise, valid_labels)

            # Print progress
            print(f"{epoch+1}/{n_epochs} [{i+1}/{int(dataset.shape[0] / n_batch)}] d_loss: {d_loss[0]:.4f}, g_loss: {g_loss:.4f}")

        # Save model checkpoints
        if (epoch + 1) % save_interval == 0:
            generator.save(f"/content/drive/MyDrive/blonde_generator_epoch_{epoch+1}.keras")
            discriminator.save(f"/content/drive/MyDrive/blonde_discriminator_epoch_{epoch+1}.keras")
            gan.save(f"/content/drive/MyDrive/blonde_gan_epoch_{epoch+1}.keras")

def main():
    if blonde_images.size == 0:
        print("No images were loaded. Exiting.")
        return

    np.random.shuffle(blonde_images)

    latent_dim = 100
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

    discriminator.trainable = False
    gan = Sequential([generator, discriminator])
    gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

    train_gan(gan, generator, discriminator, blonde_images, latent_dim)

if __name__ == "__main__":
    main()
