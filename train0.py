import os
from PIL import Image
import tensorflow as tf
from keras.layers import RandomRotation, RandomContrast

### main0.py prepares dataset by resizing as 299x299, augmentation, and splitting images into train/test/val folders
### it does so for each class (folder name). You have to go through all of them separately.

cls = "test" # d h n p s
path_base = "C:/AI/pythonProject/data/"
path_pre = path_base + "pre/" + cls + "/"
path_post = path_base + "post/" + cls + "/"
path_train = path_base + "train/" + cls + "/"
path_test = path_base + "test/" + cls + "/"
path_val = path_base + "val/" + cls + "/"

# def resize(): # resize images from path_pre and write to path_post
#     dirs = os.listdir(path_pre)
#     for item in dirs:
#         if os.path.isfile(path_pre + item):
#             if os.path.isfile(path_post + item):
#                 continue
#             else:
#                 print(item)
#                 im = Image.open(path_pre + item)
#                 # f, e = os.path.splitext(path_pre + item)
#                 imResize = im.resize((299, 299))
#                 imResize.save(path_post + item, 'JPEG', quality=90)
#
# resize()

# print("Processing " + cls + " class")
# augment_layers = tf.keras.Sequential([
#     RandomRotation(factor=(-0.05, 0.05)),
#     RandomContrast(factor=0.1),
# ])
#
# def augment():
#     dirs = os.listdir(path_pre)
#     for item in dirs:
#         filename = os.path.splitext(item)[0]
#         extension = os.path.splitext(item)[1]
#         if os.path.isfile(path_post + filename + "_" + extension):
#             continue
#         else:
#             print(filename)
#             im = Image.open(path_pre + item)
#             im2 = augment_layers(im)
#             h = int(im.height * 0.7)
#             w = int(im.width * 0.7)
#
#             im3 = tf.keras.layers.CenterCrop(h, w)(im2)
#             imResize = tf.image.resize(im3, (299, 299))
#             tf.keras.utils.save_img(x=imResize, path=path_post + filename + "_" + extension, file_format='JPEG')

# augment()


# def makedir(path):
#     try:
#         os.makedirs(path)
#     except:
#         print("folder already exists")

# makedir(path_train)
# makedir(path_test)
# makedir(path_val)

# Shuffle images and split into 800:200:200 and save in train/test/val folders
# filelist = os.listdir(path_post)
# num_files = len(filelist)
# print(num_files)

# import random
# random.shuffle(filelist)

# print("After shuffle")
# for i in range(10):
#     print(filelist[i])

# if num_files > 1200:
#     num_files = 1200
# list_train = filelist[:800]
# list_test = filelist[800:1000]
# list_val = filelist[1000:num_files]
# print(len(list_train))
# print(len(list_test))
# print(len(list_val))

# import shutil
# for i in range(len(list_train)):
#     file1 = path_post + list_train[i]
#     file2 = path_train + list_train[i]
#     shutil.move(file1, file2)
#
# for i in range(len(list_test)):
#     file1 = path_post + list_test[i]
#     file2 = path_test + list_test[i]
#     shutil.move(file1, file2)
#
# for i in range(len(list_val)):
#     file1 = path_post + list_val[i]
#     file2 = path_val + list_val[i]
#     shutil.move(file1, file2)

# path_test2 = path_base + "test/"
# dirs = os.listdir(path_pre)
# filelist = []
# for item in dirs:
#     filename = os.path.splitext(item)[0]
#     extension = os.path.splitext(item)[1]
#     filelist.append(filename)
# print(filelist)
