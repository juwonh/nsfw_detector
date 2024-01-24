from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from time import time

# Slow down training deeper into dataset
def schedule(epoch):
    if epoch < 3: # initially 6, but too slow
        # Warmup model first
        return .01
    elif epoch < 7:
       return .005
    elif epoch < 15:
        return .001
    elif epoch < 20:
        return .0005
    elif epoch < 25:
        return .0001
    elif epoch < 30:
        return .00005
    elif epoch < 35:
        return .0000032
    else:
        return .0000009

def schedule2(epoch):
    if epoch < 1:
        return .0005
    elif epoch < 3:
        return .0001
    elif epoch < 10:
        return .00005
    elif epoch < 15:
        return .00001
    elif epoch < 20:
        return .000005
    elif epoch < 30:
        return .0000032
    else:
        return .0000009
def make_callbacks(weights_file):
    # checkpoint
    filepath = weights_file
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # Update info
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    # learning rate schedule
    lr_scheduler = LearningRateScheduler(schedule)

    # all the goodies
    return [lr_scheduler, checkpoint, tensorboard]

def make_callbacks2(weights_file):
    # checkpoint
    filepath = weights_file
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # Update info
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    # learning rate schedule
    lr_scheduler = LearningRateScheduler(schedule2)

    # all the goodies
    return [lr_scheduler, checkpoint, tensorboard]