import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import subprocess


print(tf.__version__)


def load_fashion_mnist_data(data_dir):
    """Loads the Fashion-MNIST dataset.

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    License:
        The copyright for Fashion-MNIST is held by Zalando SE.
        Fashion-MNIST is licensed under the [MIT license](
        https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).

    """
    base = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    data_url = []
    data_path = []

    for fname in files:
        data_url.append(os.path.join(base, fname))
        data_path.append(os.path.join(data_dir, fname))
    print('data url: {}'.format(data_url))
    print('data path: {}'.format(data_path))

    with gzip.open(data_path[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(data_path[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(data_path[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(data_path[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return x_train, y_train, x_test, y_test


def load_data(data_dir, plot=False):
    train_images, train_labels, test_images, test_labels = load_fashion_mnist_data(data_dir)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # show info of data
    print('x_train: ', train_images.shape)
    print('y_train: ', train_labels.shape)
    print('x_test: ', test_images.shape)
    print('y_test: ', test_labels.shape)
    print('class names: ', class_names)

    if plot:
        # show an example of image
        plt.figure()
        plt.imshow(train_images[0])
        plt.colorbar()
        plt.grid(False)
        plt.show()

    # normalization, scale the values to a range of 0 to 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if plot:
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[train_labels[i]])
        plt.show()

    return np.reshape(train_images, (-1, 28, 28, 1)), train_labels, np.reshape(test_images, (-1, 28, 28, 1)), test_labels, class_names


def build_model():
    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=(28, 28, 1), filters=8, kernel_size=3,
                            strides=2, activation='relu', name='Conv1'),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
    ])
    print(model.summary())
    return model


def train(model, X_train, y_train, X_test, y_test, epochs=5):
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('\nTest accuracy: {}'.format(test_acc))
    return model


def model_save(save_path, version='v_0'):
    # if os.path.isdir(save_path):
    #     print('\nAlready saved a model, cleaning up\n')
    save_path = os.path.join(save_path, version)

    tf.saved_model.simple_save(
        keras.backend.get_session(),
        save_path,
        inputs={'input_image': model.input},
        outputs={t.name: t for t in model.outputs})

    print('\nSaved model:', save_path)



if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels, class_names = load_data('../dataset/', plot=False)
    model = build_model()
    model = train(model, train_images, train_labels, test_images, test_labels, epochs=2)
    model_save('model/pb', version='v_0')

