import os
from keras.models import load_model

# path to the trained models
model_file_path = os.path.join('model', 'cnn_model.h5')

model = load_model(model_file_path)
model.summary()

# path to the dataset
dataset_dir_path = "../dataset/images"

img_path = os.path.join(dataset_dir_path, '1500', 'mw00001.jpg')

from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(96, 96))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)

import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()

from keras import models

layer_outputs = [layer.output for layer in model.layers[:6]]

activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
plt.show()