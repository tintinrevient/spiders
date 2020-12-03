import os
from keras.models import load_model
from keras import models
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import cv2

# path to the dataset
dataset_dir_path = "../dataset/images"

# path to the trained models
model_file_path = os.path.join('model', 'cnn_model.h5')

# load the model
model = load_model(model_file_path)
model.summary()


def get_img_tensor(img_dir, img_name, show_image):

    # get the image path
    img_path = os.path.join(dataset_dir_path, img_dir, img_name)

    # load the image
    img = image.load_img(img_path, target_size=(96, 96))

    # get the image tensor
    img_tensor = image.img_to_array(img)
    # expand the dimensions
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # normalize
    img_tensor /= 255.

    print('Image shape', img_tensor.shape)

    if show_image:
        plt.imshow(img_tensor[0])
        plt.show()

    return img_tensor


def visualize_activations(img_dir, img_name, num_of_layers, images_per_row):

    # get the layer names and outputs
    layer_names = []
    layer_outputs = []
    for layer in model.layers[:num_of_layers]:
        layer_names.append(layer.name)
        layer_outputs.append(layer.output)

    # ceates a model that will return these outputs, given the model input
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    # get the image tensor
    img_tensor = get_img_tensor(img_dir, img_name, False)

    # return a list of numpy arrays
    layer_activations = activation_model.predict(img_tensor)

    # display all the feature maps
    for layer_name, layer_activation in zip(layer_names, layer_activations):
        # this is the number of features in the feature map
        n_features = layer_activation.shape[-1]

        # the feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        n_rows = n_features // images_per_row

        display_grid = np.zeros((size * n_rows, images_per_row * size))

        for row in range(n_rows):
            for col in range(images_per_row):
                channel_image = layer_activation[0, :, :, row * images_per_row + col]

                # post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= (channel_image.std() + 1e-5)
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                display_grid[row * size: (row + 1) * size, col * size: (col + 1) * size] = channel_image

        # display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

        # display
        # plt.show()

        # save
        plt.savefig(os.path.join('pix', 'activation_' + layer_name + '.png'), bbox_inches='tight', dpi=192)


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


def generate_pattern(layer_name, filter_index, show_image, size=96):

    # build a loss function that maximizes the activation of the nth filter of the layer considered
    layer_output = model.get_layer(layer_name).output

    loss = K.mean(layer_output[:, :, :, filter_index])

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])

    # we start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

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

    if show_image:
        plt.imshow(img)
        plt.show()

    return img


def visualize_filters(layer_name, n_filters, images_per_row, size=64, margin=5):

    n_cols = images_per_row
    n_rows = n_filters // n_cols

    # this is an empty (black) image where we will store our results
    results = np.zeros((n_rows * size + (n_rows - 1) * margin, n_cols * size + (n_cols - 1) * margin, 3))

    for row in range(n_rows):
        for col in range(n_cols):
            print("Row", row, 'Col', col)

            # generate the pattern for filter `row + (col * 8)` in `layer_name`
            filter_img = generate_pattern(layer_name, row * images_per_row + col, False, size=size)

            # put the result in the square `(row, col)` of the results grid
            horizontal_start = row * size + row * margin
            horizontal_end = horizontal_start + size
            vertical_start = col * size + col * margin
            vertical_end = vertical_start + size

            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    # display the results grid
    scale = 1. / size
    plt.figure(figsize=(scale * results.shape[1], scale * results.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(results.astype(np.uint8), aspect='auto')

    # display
    # plt.show()

    # save
    plt.savefig(os.path.join('pix', 'filters_' + layer_name + '.png'), bbox_inches='tight', dpi=192)


def visualize_heatmap(img_dir, img_name, last_conv_layer_name, n_filters, show_heatmap):
    # get the image tensor
    img_tensor = get_img_tensor(img_dir, img_name, False)

    pred = model.predict(img_tensor)
    class_index = np.argmax(pred)

    class_output = model.output[:, class_index]

    last_conv_layer = model.get_layer(last_conv_layer_name)

    grads = K.gradients(class_output, last_conv_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([img_tensor])

    for i in range(n_filters):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    if show_heatmap:
        plt.matshow(heatmap)
        plt.show()

    # we use cv2 to load the original image
    # get the image path
    img_path = os.path.join(dataset_dir_path, img_dir, img_name)
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
    cv2.imwrite(os.path.join('pix', 'heatmap_' + img_name), superimposed_img)


if __name__ == "__main__":

    # show image
    # get_img_tensor('1500', 'mw00001.jpg', True)

    # visualize activations
    # visualize_activations('1500', 'mw00001.jpg', 6, 16)

    # visualize filters
    # generate_pattern('conv2d', 31, True)
    # visualize_filters('conv2d', 32, 16)
    # visualize_filters('conv2d_1', 128, 16)

    # visualize heatmap
    visualize_heatmap('1500', 'mw00001.jpg', 'conv2d_1', 128, True)

