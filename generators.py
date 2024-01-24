import os
from keras.preprocessing.image import ImageDataGenerator
import constants

train_datagen = ImageDataGenerator(
    rescale=1. / 255
)

# Validation data should not be modified
validation_datagen = ImageDataGenerator(
    rescale=1. / 255
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255
)

train_dir = os.path.join(constants.BASE_DIR, 'train')
val_dir = os.path.join(constants.BASE_DIR, 'val')
test_dir = os.path.join(constants.BASE_DIR, 'test')


def create_generators(height, width):
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(height, width),
        class_mode='categorical',
        batch_size=constants.GENERATOR_BATCH_SIZE
    )

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(height, width),
        class_mode='categorical',
        batch_size=constants.GENERATOR_BATCH_SIZE
    )

    return [train_generator, validation_generator]


def create_testdata(height, width):
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(height, width),
        class_mode='categorical',
        batch_size=constants.GENERATOR_BATCH_SIZE,
        shuffle=False,
    )
    return test_generator
