import pickle

import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import os
from PIL import Image
from tqdm import tqdm

#model creation
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model, GlobalMaxPooling2D()
])

# print(model.summary())


# Feature Extraction
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocess_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocess_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# image path name and appending in the list

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

print(np.array(feature_list).shape)

#storing in pickle file
pickle.dump(feature_list, open('embedding.pkl', 'wb'))
pickle.dump(filenames, open('filemanes.pkl', 'wb'))

