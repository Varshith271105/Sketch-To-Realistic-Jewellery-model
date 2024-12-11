# Mount Google Drive to access dataset and save model checkpoints
from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dropout, LeakyReLU, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os
import numpy as np

# Define paths for images and model save location
sketch_path = '/content/drive/MyDrive/V_sk'  # Replace with your sketch folder path
realistic_path = '/content/drive/MyDrive/V_dataset'  # Replace with your realistic images folder path
model_save_dir = '/content/drive/MyDrive/V_models'
checkpoint_dir = '/content/drive/MyDrive/V_checkpoints'

os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Load and preprocess images to 1280x1280 with 3 channels
def load_images(path):
    images = []
    for img_file in sorted(os.listdir(path)):
        img = tf.io.read_file(os.path.join(path, img_file))
        img = tf.image.decode_jpeg(img, channels=3)  # Force 3 channels (RGB)
        img = tf.image.resize(img, (1280, 1280)) / 127.5 - 1  # Normalize to [-1, 1]
        images.append(img)
    return tf.data.Dataset.from_tensor_slices(images).batch(1)

# Reload sketch and realistic images as paired dataset with updated function
sketch_images = load_images(sketch_path)
realistic_images = load_images(realistic_path)
dataset = tf.data.Dataset.zip((sketch_images, realistic_images))

# Adjust U-Net Generator to 1280x1280 input
def unet_generator():
    inputs = Input(shape=[1280, 1280, 3])
    down_stack = [
        Conv2D(64, 4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)),
        Conv2D(128, 4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)),
        Conv2D(256, 4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)),
        Conv2D(512, 4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)),
        Conv2D(512, 4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)),
        Conv2D(512, 4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)),
    ]
    up_stack = [
        Conv2DTranspose(512, 4, strides=2, padding='same', activation='relu'),
        Conv2DTranspose(512, 4, strides=2, padding='same', activation='relu'),
        Conv2DTranspose(256, 4, strides=2, padding='same', activation='relu'),
        Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu'),
        Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu'),
        Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu'),  # Additional layer
        Conv2DTranspose(16, 4, strides=2, padding='same', activation='relu'),  # Additional layer
    ]

    # Down-sampling
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    # Up-sampling with skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])

    x = Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(x)
    return Model(inputs=inputs, outputs=x)

# Adjust PatchGAN Discriminator to 1280x1280 input
def patchgan_discriminator():
    inputs = Input(shape=[1280, 1280, 3])
    target = Input(shape=[1280, 1280, 3])
    x = Concatenate()([inputs, target])
    x = Conv2D(64, 4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(1, 4, strides=1, padding='same')(x)
    return Model(inputs=[inputs, target], outputs=x)

# Losses and optimizers
generator = unet_generator()
discriminator = patchgan_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Define losses
def discriminator_loss(real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return gan_loss + (100 * l1_loss)

# Training step
@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_fake_output = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_fake_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_fake_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    return gen_loss, disc_loss

# Initialize Checkpoint Manager
checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator,
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

# Restore the latest checkpoint if available
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print("Checkpoint restored from:", checkpoint_manager.latest_checkpoint)
else:
    print("No checkpoint found. Training from scratch.")

# Training Loop with Checkpoints
def train(dataset, epochs):
    start_epoch = int(checkpoint_manager.latest_checkpoint.split('-')[-1]) if checkpoint_manager.latest_checkpoint else 0
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, (input_image, target) in enumerate(dataset):
            gen_loss, disc_loss = train_step(input_image, target)
            if step % 10 == 0:
                print(f"Step {step}, Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}")

        # Save a checkpoint every epoch
        checkpoint_manager.save()
        print(f"Checkpoint saved for epoch {epoch + 1}")

        # Save generator model every 5 epochs
        if (epoch + 1) % 5 == 0:
            generator.save(f"{model_save_dir}/generator_epoch_{epoch + 1}.h5")
            print(f"Generator model saved at epoch {epoch + 1}")

            # Predict and display an example
            example_sketch = list(dataset.take(1))[0][0]
            prediction = generator(example_sketch, training=False)

            plt.figure(figsize=(15, 5))
            display_list = [example_sketch[0], prediction[0]]
            title = ['Input Sketch', 'Generated Image']

            for i in range(2):
                plt.subplot(1, 2, i+1)
                plt.title(title[i])
                plt.imshow((display_list[i] * 0.5 + 0.5))  # Rescale to [0, 1] for display
                plt.axis('off')
            plt.show()

# Run the training process
train(dataset, epochs=100)