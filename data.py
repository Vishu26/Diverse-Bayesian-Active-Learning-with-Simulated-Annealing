from keras.datasets import mnist, fashion_mnist, cifar10
import numpy as np
from tensorflow.keras.utils import to_categorical

def get_data(data='mnist'):
    
    if data=='mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train, y_train, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)
        return X_train, y_train, X_test, y_test

    elif data=='fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train, y_train, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)
        return X_train, y_train, X_test, y_test
    
    elif data=='cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train, y_train, X_test, y_test = preprocess_data_cifar10(X_train, y_train, X_test, y_test)
        return X_train, y_train, X_test, y_test
    
    else:
        raise "Invalid dataset"

def get_noisy_data(data='mnist'):

    if data=='mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train, y_train, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)
        X_noisy_train = 0.5*X_train[:30000, :] + 0.5*np.random.normal(0, 1, (30000, 28, 28, 1))
        X_noisy_train = np.clip(X_noisy_train, 0, 1)
        X_noisy_train = np.concatenate((X_noisy_train, X_train[30000:]), axis=0)
        return X_noisy_train, y_train, X_test, y_test

    elif data=='fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train, y_train, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)
        X_noisy_train = 0.5*X_train[:30000, :] + 0.5*np.random.normal(0, 1, (30000, 28, 28, 1))
        X_noisy_train = np.clip(X_noisy_train, 0, 1)
        X_noisy_train = np.concatenate((X_noisy_train, X_train[30000:]), axis=0)
        return X_noisy_train, y_train, X_test, y_test
    
    elif data=='cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train, y_train, X_test, y_test = preprocess_data_cifar10(X_train, y_train, X_test, y_test)
        X_noisy_train = 0.5*X_train[:30000, :] + 0.5*np.random.normal(0, 1, (30000, 32, 32, 1))
        X_noisy_train = np.clip(X_noisy_train, 0, 1)
        X_noisy_train = np.concatenate((X_noisy_train, X_train[30000:]), axis=0)
        return X_noisy_train, y_train, X_test, y_test
    
    else:
        raise "Invalid dataset"   


def preprocess_data(X_train, y_train, X_test, y_test):

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = X_train/255.0
    X_test = X_test/255.0

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test

def preprocess_data_cifar10(X_train, y_train, X_test, y_test):

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 3)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 3)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = np.sum(X_train/3, axis=3, keepdims = True)
    X_test = np.sum(X_test/3, axis=3, keepdims = True)
    
    X_train = (X_train - 128) / 128
    X_test = (X_test - 128) / 128

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test