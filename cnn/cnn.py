from keras import models
from keras import layers
from keras_preprocessing import image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


# number of training epochs
epochs = 3
# batch size
batch_size = 20
# number of classes
num_classes = 3

# path to the dataset
dataset_dir_path = "../dataset/images"

# path to the trained models
model_dirname = 'model'
if not os.path.exists(model_dirname):
    os.mkdir(model_dirname)

model_filename = 'cnn_model.h5'
model_path = os.path.join(model_dirname, model_filename)

# path to the trained models' weights
weights_dirname = 'weights'
if not os.path.exists(weights_dirname):
    os.mkdir(weights_dirname)

weights_filename = 'weights_model.npy'
weights_path = os.path.join(weights_dirname, weights_filename)

def preprocess():

    train_data_generator = image.ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    training_generator = train_data_generator.flow_from_directory(
        directory=dataset_dir_path,
        target_size=(96, 96),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    validation_generator = train_data_generator.flow_from_directory(
        directory=dataset_dir_path,
        target_size=(96, 96),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    return training_generator, validation_generator


def plot_history(history):
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training and validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Training and validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()


def train():
    training_generator, validation_generator = preprocess()

    model = models.Sequential([
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(96, 96, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(units=512, activation='relu'),
        layers.Dense(units=num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=training_generator.n / batch_size,
                                  epochs=epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_generator.n / batch_size,
                                  verbose=1)

    plot_history(history)

    loss, accuracy = model.evaluate_generator(validation_generator)
    print('\nTest accuracy:', accuracy)

    # save the model
    model.save(model_path)

    # save the model's weights
    np.save(weights_path, model.get_weights())


def predict(avg_year, img_filename):

    datagen = image.ImageDataGenerator(rescale=1. / 255)

    testgen = datagen.flow_from_directory(directory=dataset_dir_path,
                                          target_size=(96, 96),
                                          batch_size=batch_size,
                                          class_mode="categorical")

    # class indices dictionary with the mapping: class_name -> class_index
    class_indices = testgen.class_indices
    class_names = []
    for i, class_name in enumerate(class_indices):
        class_names.append(class_name)

    # load the test image
    img_path = os.path.join(dataset_dir_path, str(avg_year), img_filename)
    img = image.load_img(img_path, target_size=(96, 96))
    img_array = image.img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)
    img_array = img_array / 255.

    # load the model and predict the label
    model = models.load_model(model_path)
    pred = model.predict(img_array)

    predicted_label = class_names[np.argmax(pred)]
    predicted_prob = np.max(pred)

    true_label = str(avg_year)

    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    # plot the image
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                         100 * predicted_prob,
                                         true_label), color=color)
    plt.show()


if __name__ == "__main__":

    # train the model
    train()

    # predict the label given the image index
    # predict(16860) # 83% eyewear
    # predict(22256) # 60% shoes
    # predict(49764) # 100% bottomwear
