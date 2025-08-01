import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from PIL import Image

# Set dataset directories
dataset_dir = "D:/SkinDiseaseDetection/backend/model/data"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")
val_dir = os.path.join(dataset_dir, "validation")

# Verify dataset exists
for folder in [train_dir, test_dir, val_dir]:
    if not os.path.exists(folder) or not os.listdir(folder):
        print(f"⚠ Warning: '{folder}' folder not found or empty!")

# Function to remove corrupt images
def remove_corrupt_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                img = Image.open(os.path.join(root, file))
                img.verify()
            except Exception as e:
                print(f"Removing corrupt image: {file}")
                os.remove(os.path.join(root, file))

# Clean dataset
remove_corrupt_images(train_dir)
remove_corrupt_images(test_dir)
remove_corrupt_images(val_dir)

# Image augmentation and data loading
img_size = (224, 224)
batch_size = 16

data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = data_gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
test_generator = data_gen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
val_generator = data_gen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

# Load MobileNetV2
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(train_generator.num_classes, activation='softmax')(x)

# Compile model
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
epochs = 20
history = model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# Save model
model.save("D:/SkinDiseaseDetection/backend/model/skin_disease_model.h5")
print("✅ Model training complete and saved!")
