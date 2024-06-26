# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, LeakyReLU, Conv2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define paths
PROCESSED_IMAGES_PATH = '/content/drive/MyDrive/ProcessedImages/'
BLONDE_IMAGES_FILE = os.path.join(PROCESSED_IMAGES_PATH, 'blonde_images.npy')

# Load preprocessed images
blonde_images = np.load(BLONDE_IMAGES_FILE)
print(f"Loaded {blonde_images.shape[0]} images for training.")

# Gradient penalty loss for WGAN-GP
def gradient_penalty(discriminator, real_images, fake_images, batch_size):
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_images + (1 - alpha) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated)

    grads = tape.gradient(pred, interpolated)
    grads = tf.reshape(grads, [batch_size, -1])
    norm = tf.norm(grads, axis=1)
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

# Define GAN components for WGAN-GP
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
    model.add(Dense(1))
    return model

# Loss functions for WGAN-GP
def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

def discriminator_loss(real_output, fake_output, gp, gradient_penalty_weight):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + gradient_penalty_weight * gp

# Training step for WGAN-GP
@tf.function
def train_step(generator, discriminator, images, batch_size, latent_dim, generator_optimizer, discriminator_optimizer, gradient_penalty_weight):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gp = gradient_penalty(discriminator, images, generated_images, batch_size)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output, gp, gradient_penalty_weight)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train_gan(generator, discriminator, dataset, latent_dim, n_epochs=100, batch_size=128, save_interval=10, gradient_penalty_weight=10.0):
    generator_optimizer = Adam(learning_rate=0.0001, beta_1=0.5)
    discriminator_optimizer = Adam(learning_rate=0.0001, beta_1=0.5)

    for epoch in range(n_epochs):
        print(f"Starting epoch {epoch+1}/{n_epochs}")
        for i in range(int(dataset.shape[0] / batch_size)):
            print(f"Training epoch {epoch+1}/{n_epochs}, batch {i+1}/{int(dataset.shape[0] / batch_size)}")

            idx = np.random.randint(0, dataset.shape[0], batch_size)
            real_images = dataset[idx]
            gen_loss, disc_loss = train_step(generator, discriminator, real_images, batch_size, latent_dim, generator_optimizer, discriminator_optimizer, gradient_penalty_weight)

            print(f"Generator loss: {gen_loss:.4f}, Discriminator loss: {disc_loss:.4f}")

        if (epoch + 1) % save_interval == 0:
            generator.save(f"/content/drive/MyDrive/blonde_generator_epoch_{epoch+1}.keras")
            discriminator.save(f"/content/drive/MyDrive/blonde_discriminator_epoch_{epoch+1}.keras")

def main():
    if blonde_images.size == 0:
        print("No images were loaded. Exiting.")
        return

    np.random.shuffle(blonde_images)

    latent_dim = 100
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()

    train_gan(generator, discriminator, blonde_images, latent_dim, n_epochs=100, batch_size=128)

if __name__ == "__main__":
    main()
