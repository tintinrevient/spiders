import os
from keras.models import load_model
from keras_preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


# path to the trained models
model_file_path = os.path.join('model', 'cnn_model.h5')

model = load_model(model_file_path)
model.summary()

# path to the dataset
dataset_dir_path = "../dataset/images"

img_path = os.path.join(dataset_dir_path, '1500', 'mw00001.jpg')

img = image.load_img(img_path, target_size=(96, 96))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.

pred = model.predict(x)
print(np.argmax(pred))

aimed_output = model.output[:, 1]

last_conv_layer = model.get_layer('conv2d_1')

grads = K.gradients(aimed_output, last_conv_layer.output)[0]

# this is a vector of shape (128,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))
# print(pooled_grads.shape)

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

# we multiply each channel in the feature map array
# by "how important this channel is" with regard to the aimed class
for i in range(128):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

import cv2

# we use cv2 to load the original image
img = cv2.imread(img_path)

# we resize the heatmap to have the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# we convert the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# we apply the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4 here is a heatmap intensity factor
superimposed_img = heatmap * 0.4 + img

# save the image to disk
cv2.imwrite('./pix/heatmap.jpg', superimposed_img)