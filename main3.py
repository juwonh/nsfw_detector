from keras.models import Sequential, Model, load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import generators
import constants

### main3.py runs the model with the test dataset and produces confusion matrix

height = constants.SIZES['basic']
width = height

test_generator = generators.create_testdata(height, width)
y_true = test_generator.classes
filelist = test_generator.filenames
# print(filelist)

# model = load_model("saved.model.h5")
# weights_file = "weights.best_1.hdf5"

# model = load_model("saved.model.keras")
# weights_file = "weights.tuned_1.hdf5"

model = load_model("model_1.keras") ## This is the best: 96 % accuracy
weights_file = "weights.best_1_1.hdf5"
# model.load_weights(weights_file)

model_preds = model.predict(test_generator)

print("model_preds")
# print(model_preds)
print(model_preds.shape)

probs = []

for i, single_preds in enumerate(model_preds):
    p = 0
    cls = 0
    for j, pred in enumerate(single_preds):
        if pred > p:
            p = pred
            cls = j

    probs.append(cls)

print("probs")
print(probs)

cm = confusion_matrix(y_true, probs)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='g')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.axis('off')
plt.show()
