import json
import gzip
import os
import requests
import time

import numpy as np
import grpc
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib.util import make_tensor_proto
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


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

    return np.reshape(train_images, (-1, 28, 28, 1)), train_labels, np.reshape(test_images, (
    -1, 28, 28, 1)), test_labels, class_names


def test_http_rest(test_images):
    data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
    print('Data: {} ... {}'.format(data[:50], data[len(data) - 52:]))

    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/models/mnist/:predict', data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']


def test_grpc(test_images, model_name, host='localhost', port=8500, signature_name='serving_default'):
    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # # Read an image
    # data = cv2.resize(cv2.imread(image, 0), (28, 28))
    # # data = imread(image)
    # data = data.astype(np.float32)
    # print(data)

    start = time.time()

    # Call classification model to make prediction on the image
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    # request.inputs['input_image'].CopyFrom(make_tensor_proto(test_images[:3].astype(np.float32), shape=[-1, 28, 28, 1]))
    request.inputs['input_image'].CopyFrom(make_tensor_proto(test_images[:5].astype(np.float32)))

    result = stub.Predict(request, 10.0)

    end = time.time()
    time_diff = end - start

    # Reference:
    # How to access nested values
    # https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
    print(result)
    print('time elapased: {}'.format(time_diff))
    return np.asarray(result.outputs['Softmax/Softmax:0'].float_val)


def show(idx, title):
    plt.figure()
    plt.imshow(test_images[idx].reshape(28, 28))
    plt.axis('off')
    plt.title('\n\n{}'.format(title), fontdict={'size': 16})
    plt.show()



if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels, class_names = load_data('../dataset/', plot=False)
    # test_http_rest(test_images)
    predictions =  test_grpc(test_images, 'mnist')

    predictions = predictions.reshape((-1, 10))

    import random
    #
    # rando = random.randint(0, len(test_images) - 1)
    # show(rando, 'An Example Image: {}'.format(class_names[test_labels[rando]]))

    for index in range(0, predictions.shape[0]):
        show(index, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
            class_names[np.argmax(predictions[index])], test_labels[index], class_names[np.argmax(predictions[index])], test_labels[index]))