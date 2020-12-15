import os
from keras.models import load_model
import tensorflow as tf
import keras
import numpy as np


# path to the trained models
model_file_path = os.path.join('model', 'cnn_model.h5')

model = load_model(model_file_path)
model.summary()

# layer name
layer_name = 'conv2d_1'

# Set up a model that returns the activation values for our target layer
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

img_width = 96
img_height = 96


def deprocess_image(img):

    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")

    return img


# The "loss" we will maximize is simply the mean of the activation of a specific filter in our target layer.
# To avoid border effects, we exclude border pixels.
def compute_loss(input_image, filter_index):

    activation = feature_extractor(input_image)

    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]

    return tf.reduce_mean(filter_activation)


# Our gradient ascent function simply computes the gradients of the loss above with regard to the input image,
# and update the image so as to move it towards a state that will activate the target filter more strongly.
@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):

    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)

    # Compute gradients.
    grads = tape.gradient(loss, img)

    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads

    return loss, img


def initialize_image():

    # We start from a gray image with some random noise
    # For floats, the default range is [0, 1)
    img = tf.random.uniform(shape=(1, img_width, img_height, 3))

    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    # return (img - 0.5) * 0.25
    return img * 20 + 128.


# run gradient ascent for 40 steps
def visualize_filter(filter_index):

    # We run gradient ascent for 40 steps
    iterations = 40

    learning_rate = 10.0
    img = initialize_image()

    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())

    return loss, img


from IPython.display import Image, display

loss, img = visualize_filter(0)
keras.preprocessing.image.save_img("0.png", img)

display(Image("0.png"))