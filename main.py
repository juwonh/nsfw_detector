import os
import sys
import numpy as np
from os.path import exists
from keras.preprocessing import image
from keras.models import load_model
### This one is executable for classifying one image into 5 classes (0,1,2: okay, 3,4: nsfw)

def get_model(model_path):
    if model_path is None or not exists(model_path):
        raise ValueError("saved_model_path must be the valid directory of a saved model to load.")
    model = load_model(model_path, compile=False)
    return model

def get_img_299(img_path):
    if os.path.exists(img_path):
        try:
            print("loading image: ", img_path)
            img = image.load_img(img_path, target_size=(299,299))
            img = image.img_to_array(img)
            img /= 255
        except Exception as ex:
            print("Image load failure: ", img_path, ex)
    return img

def if_an_image_is_harmful(model, image_array):
    single_preds = model.predict(image_array)
    # print(single_preds)
    p = 0
    cls = 0
    harmful = False
    for j, pred in enumerate(single_preds[0]):
        if pred > p:
            p = pred
            cls = j
    # print(cls)
    if( cls > 2 ):
        harmful = True
    return harmful

def main():
    args = sys.argv

    # img_path = "C:\\AI\\pythonProject\\image\\test\\0\\1aea5.jpg"
    if(len(args) > 1): # first argument should be the image path
        img_path = args[1]
        loaded_image = []
        img = get_img_299(img_path)
        loaded_image.append(img)

        model_path = "model_1.keras"  # This one is the best: 96 % accuracy
        print("loading model: ", model_path)
        model = get_model(model_path)

        harmful = if_an_image_is_harmful(model, np.asarray(loaded_image))
        print("harmful :", harmful)

    else:
        print("!!! first argument should be the image path like below !!!")
        print("python main.py \"C:\\AI\\pythonProject\\image\\test\\0\\1aea5.jpg\"")
        print("!!!----------------------------------------------------!!!")



### How to run command line executable
### python main.py "C:\\AI\\pythonProject\\image\\test\\0\\1aea5.jpg"
if __name__ == "__main__":
    main()