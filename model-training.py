import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image

# Validate Images Before Loading
def validate_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    img.verify()  # Validate image
            except Exception as e:
                print(f"Removing corrupted image: {img_path}")
                os.remove(img_path)

# Correct dataset paths
train_dir = 'C:/Users/shrut/Downloads/archive/chest_xray/chest_xray/train'
val_dir = val_dir = 'C:/Users/shrut/Downloads/archive/chest_xray/chest_xray/test'


# Validate datasets before loading
validate_images(train_dir)
validate_images(val_dir)

# Data Preprocessing
datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the Model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Save the Model
model.save('pneumonia_predictor.h5')

print("Model training complete and saved.")
