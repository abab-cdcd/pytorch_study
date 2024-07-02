"""
code copied from https://github.com/cbfinn/maml/blob/master/data/miniImagenet/proc_images.py
Script for converting from csv file datafiles to a directory for each image (which is how it is loaded by MAML code)

Acquire miniImagenet from Ravi & Larochelle '17, along with the train, val, and test csv files. Put the
csv files in the miniImagenet directory and put the images in the directory 'miniImagenet/images/'.
Then run this script from the miniImagenet directory:
    cd data/miniImagenet/
    python proc_images.py
"""

from __future__ import print_function
import csv
import glob
import os

from PIL import Image

# path_to_images = 'images/'

# all_images = glob.glob(path_to_images + '*')

# Resize images
train_data_folder = './train/'

train_character_folders = [os.path.join(train_data_folder, family,character) \
                           for family in os.listdir(train_data_folder)\
                           if os.path.isdir(os.path.join(train_data_folder, family)) \
                     for character in os.listdir(os.path.join(train_data_folder, family))]
# character_folders = [os.path.join(data_folder, family, character) \
#                      for family in os.listdir(data_folder) \
#                      if os.path.isdir(os.path.join(data_folder, family)) \
#                      for character in os.listdir(os.path.join(data_folder, family))]
test_data_folder = './test/'
# print(train_character_folders)
test_character_folders = [os.path.join(test_data_folder, family,character) \
                          for family in os.listdir(test_data_folder) \
                          if os.path.isdir(os.path.join(test_data_folder, family)) \
                          for character in os.listdir(os.path.join(test_data_folder, family))
                          ]
for i, image_file in enumerate(train_character_folders):
    im = Image.open(image_file)
    im = im.resize((84, 84), resample=Image.LANCZOS)
    im.save(image_file)
    if i % 500 == 0:
        print(i)
for i, image_file in enumerate(test_character_folders):
    im = Image.open(image_file)
    im = im.resize((84, 84), resample=Image.LANCZOS)
    im.save(image_file)
    if i % 500 == 0:
        print(i)

