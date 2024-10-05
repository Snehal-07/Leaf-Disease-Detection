#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#### import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Reshape, LeakyReLU, Conv2DTranspose, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Paths
data_dir = r"C:\Mini project\archive (2)"

# Image Data Generator for Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split 20% for validation
)

# Load Training and Validation Data
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),  # Resize to 128x128
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build CNN Model
def build_cnn():
    model = Sequential()

    # First convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(2, 2))

    # Second convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    # Third convolutional layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(train_generator.num_classes, activation='softmax'))  # Number of classes

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

cnn_model = build_cnn()

# Train CNN Model
cnn_model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Build Generator Model
def build_generator():
    model = Sequential()

    model.add(Dense(256 * 16 * 16, activation='relu', input_dim=100))
    model.add(Reshape((16, 16, 256)))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh'))

    return model

generator1 = build_generator()  # First Generator
generator2 = build_generator()  # Second Generator

# Build Discriminator Model
def build_discriminator():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(128, 128, 3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

discriminator1 = build_discriminator()  # First Discriminator
discriminator2 = build_discriminator()  # Second Discriminator

# Combine Generator and Discriminator (GAN model)
def build_gan(generator, discriminator):
    discriminator.trainable = False

    gan_input = Input(shape=(100,))
    generated_image = generator(gan_input)

    gan_output = discriminator(generated_image)

    gan_model = Model(gan_input, gan_output)
    gan_model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
    return gan_model

# Double GAN
gan1 = build_gan(generator1, discriminator1)
gan2 = build_gan(generator2, discriminator2)

# Training the GAN
def train_gan(gan, generator, discriminator, epochs=1000, batch_size=32):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        # Get real images (from dataset)
        real_images, _ = next(train_generator)
        real_images = real_images[:half_batch]  # Select half batch
        
        # Resize batch for consistency
        real_images = real_images[:half_batch]

        # Generate fake images
        noise = np.random.normal(0, 1, (half_batch, 100))
        fake_images = generator.predict(noise)

        # Train Discriminator
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator via GAN
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.ones((batch_size, 1))

        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, valid_y)

        # Print Progress
        if epoch % 100 == 0:
            print(f'{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]')

# Train both GANs
train_gan(gan1, generator1, discriminator1, epochs=1000, batch_size=16)
train_gan(gan2, generator2, discriminator2, epochs=1000, batch_size=16)

# Generate new images using the first generator
noise = np.random.normal(0, 1, (5, 100))
gen_images = generator1.predict(noise)

# Plot generated images
plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow((gen_images[i] * 0.5) + 0.5)  # Rescale images from [-1, 1] to [0, 1]
    plt.axis('off')
plt.show()


# In[ ]:




