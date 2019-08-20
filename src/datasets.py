import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
import os
import urllib.request
from scipy.io import loadmat


# Convert to float32 and Normalize images value from [0, 255] to [0, 1].
def data_normalize(x_train, x_test):
    x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
    x_train, x_test = x_train / 255., x_test / 255.
    return x_train, x_test


class MNIST(object):
    def __init__(self, batch_size):

        # MNIST dataset parameters. Total classes (0-9 digits).
        self.num_classes = 10

        # Prepare MNIST data.
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
        x_train, x_test = data_normalize(x_train, x_test)
        y_train, y_test = np.array(y_train, np.int32), np.array(y_test, np.int32)

        # Use tf.data API to shuffle and batch data.
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        self.train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
        self.test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).prefetch(1)
        self.train_size = x_train[0]
        self.test_size = x_test[0]


class SVHN(object):
    def __init__(self, data_dir_path, batch_size):

        # Download SVHN dataset if not available
        if not os.path.isdir(data_dir_path):
            os.mkdir(data_dir_path)

        if not os.path.exists(os.path.join(data_dir_path, "train_32x32.mat")):
            print('Beginning file train_32x32.mat...')
            url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
            urllib.request.urlretrieve(url, os.path.join(data_dir_path, "train_32x32.mat"))

        if not os.path.exists(os.path.join(data_dir_path, "test_32x32.mat")):
            print('Beginning file train_32x32.mat...')
            url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
            urllib.request.urlretrieve(url, os.path.join(data_dir_path, "test_32x32.mat"))

        # SVHN dataset parameters. Total classes (0-9 digits).
        self.num_classes = 10

        # Load SVHN
        train = loadmat(os.path.join(data_dir_path, "train_32x32.mat"))
        test = loadmat(os.path.join(data_dir_path, "test_32x32.mat"))

        # Change format
        x_train, y_train = self.change_format(train)
        x_test, y_test = self.change_format(test)

        x_train, x_test = data_normalize(x_train, x_test)
        y_train, y_test = np.array(y_train, np.int32), np.array(y_test, np.int32)

        # Use tf.data API to shuffle and batch data.
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        self.train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
        self.test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).prefetch(1)
        self.train_size = x_train.shape[0]
        self.test_size = x_test.shape[0]

    @staticmethod
    def change_format(mat):
        """
        Convert X: (HWCN) -> (NHWC) and Y: [1,...,10] -> one-hot
        """
        x = mat['X'].transpose((3, 0, 1, 2))
        y = mat['y'].reshape(-1)
        y[y == 10] = 0
        y = np.eye(10)[y]
        return x, y


class CIFAR10(object):
    def __init__(self, batch_size):

        # MNIST dataset parameters. Total classes (0-9 digits).
        self.num_classes = 10

        # Prepare data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = data_normalize(x_train, x_test)
        y_train, y_test = np.array(y_train, np.int32), np.array(y_test, np.int32)

        # Use tf.data API to shuffle and batch data
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        self.train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
        self.test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).prefetch(1)
        self.train_size = x_train.shape[0]
        self.test_size = x_test.shape[0]


class STL(object):
    def __init__(self, batch_size):
        print('not yet')
