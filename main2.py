import os
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import clear_session
from keras.optimizers import SGD, Adam
from pathlib import Path
from keras.models import Sequential, Model, load_model
import generators
import constants
import callbacks

### main2.py fine-tune trained model by making more layers trainable
### but it doesn't improve false positive because the trained model already overfitted.

height = constants.SIZES['basic']
width = height

print ('Starting from last full model run')
model = load_model("model_1_2.keras")

# # Let's see it
# print('Summary')
# print(model.summary())

# Load checkpoint if one is found
weights_file = "weights.best_1_2.hdf5"
if os.path.exists(weights_file):
    print ("loading ", weights_file)
    model.load_weights(weights_file)

# Unlock a few layers deep in Inception v3
model.trainable = True
set_trainable = False
for layer in model.layers:
    if layer.name == 'conv2d_85': # 84 89 86 90 87 88 91 92 85    93
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


weights_file = "weights.best_1_3.hdf5"
# Get all model callbacks
callbacks_list = callbacks.make_callbacks2(weights_file)

print('Compile model')
# originally adam, but research says SGD with scheduler
# opt = Adam(learning_rate=0.01, amsgrad=True)
opt = SGD(momentum=.9)
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

# Get training/validation data via generators
train_generator, validation_generator = generators.create_generators(height, width)

print('Start training!')
history = model.fit(
    train_generator,
    callbacks=callbacks_list,
    epochs = 10, # =constants.TOTAL_EPOCHS,
    steps_per_epoch=constants.STEPS_PER_EPOCH,
    shuffle=True,
    workers=4,
    use_multiprocessing=False,
    validation_data=validation_generator,
    validation_steps=constants.VALIDATION_STEPS,
)

# Save it for later
print('Saving Model')
# model.save("nsfw." + str(width) + "x" + str(height) + "_tuning.h5")
model.save('model_1_3.keras')