import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('modelKidneyStone.h5')

image = cv2.imread('datasets/Tumor/Tumor- (2).jpg')

img = Image.fromarray(image, 'RGB')
img = img.resize((64, 64))
img = np.array(img)

input_img = np.expand_dims(img, axis=0)
result = model.predict(input_img)

print(result)
