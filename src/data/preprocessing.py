import os
import numpy as np
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input


IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def get_base_dir():
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )


def get_generators(use_resnet=True):
    base_dir = get_base_dir()

    data_dir = os.path.join(base_dir, "data", "raw", "brain_dataset")
    train_dir = os.path.join(data_dir, "Training")
    test_dir = os.path.join(data_dir, "Testing")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Testing directory not found: {test_dir}")

    if use_resnet:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            validation_split=0.15
        )

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

    else:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            validation_split=0.15
        )

        test_datagen = ImageDataGenerator(
            rescale=1.0 / 255
        )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        subset="training"
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        subset="validation"
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen

#PREPROCESSING FUNCTION FOR SINGLE IMAGE
def preprocess_image(image, use_resnet=True):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    if use_resnet:
        image_array = preprocess_input(image_array)
    else:
        image_array = image_array / 255.0

    return image_array
