import os
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

def deprocess_image(x):

    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')

    return x


# path to the trained models
model_file_path = os.path.join('model', 'cnn_model.h5')

model = load_model(model_file_path)
model.summary()

layer_name = 'conv2d_1'

filter_index = 1

layer_output = model.get_layer(layer_name).output

loss = K.mean(layer_output[:, :, :, filter_index])

# the call to `gradients` returns a list of tensors (of size 1 in this case)
# hence we only keep the first element -- which is a tensor.
grads = K.gradients(loss, model.input)[0]

# we add 1e-5 before dividing so as to avoid accidentally dividing by 0.
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

iterate = K.function(inputs=[model.input], outputs=[loss, grads])

import numpy as np

# loss_value, grads_value = iterate([np.zeros((1, 96, 96, 3))])

# we start from a gray image with some noise
input_img_data = np.random.random((1, 96, 96, 3)) * 20 + 128.

# run gradient ascent for 40 steps

# this is the magnitude of each gradient update
step = 1.

for i in range(40):
    # compute the loss value and gradient value
    loss_value, grads_value = iterate([input_img_data])

    # here we adjust the input image in the direction that maximizes the loss
    input_img_data += grads_value * step


img = input_img_data[0]

img = deprocess_image(img)

plt.imshow(img)
plt.show()