import os
import tensorflow as tf
print(tf.__version__)
# Load the generator model
generator = tf.keras.models.load_model('<GENERATOR_MODEL>')

# Function to generate and save images
def generate_and_save_images(generator, latent_dim, num_images, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(num_images):
        noise = tf.random.normal([1, latent_dim])
        generated_image = generator(noise, training=False)
        generated_image = (generated_image + 1) / 2.0  # Rescale to [0, 1]
        generated_image = tf.clip_by_value(generated_image, 0.0, 1.0)  # Ensure values are in [0, 1]
        
        img_path = os.path.join(output_dir, f'generated_image_{i+1}.png')
        tf.keras.preprocessing.image.save_img(img_path, generated_image[0])
        print(f'Saved: {img_path}')

# Parameters
latent_dim = 100
num_images = 10  # Number of images to generate
output_dir = 'generated_images'  # Output directory to save images

# Generate and save images
generate_and_save_images(generator, latent_dim, num_images, output_dir)
