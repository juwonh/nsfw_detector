from keras.applications import InceptionV3
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D
from keras import initializers, regularizers
from keras.models import Model
from keras.optimizers import SGD
import callbacks
import constants
import generators

### main1.py trains the InceptionV3 model with added top layers

height = 299
width = 299

conv_base = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(height, width, constants.NUM_CHANNELS)
)

conv_base.trainable = False
# print(conv_base.summary())

# Let's construct that top layer replacement
x = conv_base.output
x = AveragePooling2D(pool_size=(8, 8))(x)
x - Dropout(0.4)(x)
x = Flatten()(x)
x = Dense(256, activation='relu', kernel_initializer=initializers.he_normal(seed=None), kernel_regularizer=regularizers.l2(.0005))(x)
x = Dropout(0.5)(x)
# Essential to have another layer for better accuracy
x = Dense(128,activation='relu', kernel_initializer=initializers.he_normal(seed=None))(x)
x = Dropout(0.25)(x)
predictions = Dense(constants.NUM_CLASSES,  kernel_initializer="glorot_uniform", activation='softmax')(x)

# print('Stacking New Layers')
model = Model(inputs = conv_base.input, outputs=predictions)
# model.summary()

# Load checkpoint if one is found
import os
weights_file = "weights.best_1.hdf5"
if os.path.exists(weights_file):
    print ("loading ", weights_file)
    model.load_weights(weights_file)

weights_file = "weights.best_2.hdf5"
callbacks_list = callbacks.make_callbacks(weights_file)

print('Compile model')
# originally adam, but research says SGD with scheduler
# opt = Adam(lr=0.001, amsgrad=True)
opt = SGD(momentum=.9)
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'] #'val_acc'
)

# Get training/validation data via generators
train_generator, validation_generator = generators.create_generators(height, width)

model.save("model.keras")
print('Start training!')  ### 1 epoch takes about 10 minutes
history = model.fit(
    train_generator,
    callbacks=callbacks_list,
    epochs=10,    # constants.TOTAL_EPOCHS,
    steps_per_epoch=constants.STEPS_PER_EPOCH,
    shuffle=True,
    workers=4,
    use_multiprocessing=False,
    validation_data=validation_generator,
    validation_steps=constants.VALIDATION_STEPS
)

# Save it for later
print('Saving Model')
model.save("model.keras")