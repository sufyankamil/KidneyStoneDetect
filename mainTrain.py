import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import normalize

from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

img_directory = 'datasets/'

normal_image = os.listdir(img_directory+ 'Normal/')
stone_image = os.listdir(img_directory+ 'Stone/')


dataset = []
label = []

INPUT_SIZE = 64

#print(normal_image)
#path = "datasets/Normal/Normal- (1).jpg"
#print(path)

for i, image_name in enumerate(normal_image):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(img_directory+ 'Normal/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(stone_image):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(img_directory+ 'Stone/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)


dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)

#Reshape = (n, image_width, image_height, n_channels)

#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)


# normalize forms

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# building model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(INPUT_SIZE,INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test), shuffle=False)

model.save('modelKidneyStone.h5')

