#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import numpy as np
import os
import gc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
data_dir = r"C:\Mini project\archive (2)"

# Image Data Generator for Real Data
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),  # Reduced image size
    batch_size=8,  # Reduced batch size
    class_mode='categorical',
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)

# GAN Models
latent_dim = 100

# Generator Model
def build_generator(latent_dim=100):
    model = Sequential([
        Dense(128, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dense(64 * 64 * 3, activation='tanh'),
        Reshape((64, 64, 3))
    ])
    return model

# Discriminator Model
def build_discriminator():
    model = Sequential([
        Flatten(input_shape=(64, 64, 3)),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

# GAN Compilation
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_model = Sequential([generator, discriminator])
    gan_model.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')
    return gan_model

# Instantiate GAN Models
generator = build_generator(latent_dim)
discriminator = build_discriminator()
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
gan = build_gan(generator, discriminator)

# GAN Training Function
def train_gan(epochs=100, batch_size=8):
    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = generator.predict(noise)

        real_images, _ = next(train_generator)
        real_images = real_images[:batch_size]  # Ensure batch size consistency

        if real_images.shape[0] != batch_size:
            continue  # Skip if not enough real images

        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_images, labels_real)
        d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, labels_real)

        if epoch % 10 == 0:  # Print every 10 epochs
            print(f"Epoch {epoch} [D loss: {0.5 * np.add(d_loss_real[0], d_loss_fake[0])}] [G loss: {g_loss}]")

train_gan(epochs=100)

# Simple CNN-based Disease Classifier
def build_simple_classifier(num_classes=3):  # Adjusted to 3 classes
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),  # Fewer filters
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(32, activation='relu'),  # Fewer units
        Dense(num_classes, activation='softmax')  # Set to 3 classes
    ])
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train Disease Classifier with limited epochs
simple_classifier = build_simple_classifier(num_classes=train_generator.num_classes)
history = simple_classifier.fit(
    train_generator,
    epochs=5,  # Fewer epochs
    validation_data=validation_generator
)

# Evaluate the model's performance
val_loss, val_accuracy = simple_classifier.evaluate(validation_generator)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Prediction Function for Disease Detection
def predict_disease(image_path):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(64, 64))  # Resize to match model input
    image = img_to_array(image) / 255.0  # Scale image to match training data
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match batch format
    
    # Predict using the model
    prediction = simple_classifier.predict(image)
    detected_class = np.argmax(prediction)
    
    # Assuming class 1 represents 'Disease Detected'
    if detected_class == 1:
        return "No Disease Detected"
    else:
        return "Disease Detected"

# Test the function with an image path
image_path = r"C:/Mini project/archive (2)/Validation/Validation/Rust/8437f01fd3d20f26.jpg"
result = predict_disease(image_path)
print(result)


# Clear unused variables and free memory
del generator, discriminator, gan, simple_classifier
gc.collect()


# In[ ]:




