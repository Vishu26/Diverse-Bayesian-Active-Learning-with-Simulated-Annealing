import numpy as np
from PIL import Image
import os
import glob
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


def read_images(path='UCMerced_LandUse/Images/', shape=(256, 256, 3)):
    x_train = np.zeros((0, shape[0], shape[1], shape[2]), dtype=np.float32)
    y_train = np.zeros((0, 1), dtype=np.uint8)
    x_test = np.zeros((0, shape[0], shape[1], shape[2]), dtype=np.float32)
    y_test = np.zeros((0, 1), dtype=np.uint8)
    dirs = os.listdir(path)
    label = 0
    for i in tqdm(dirs):
        count = 0
        for pic in glob.glob(path+i+'/*.tif'):
            im = Image.open(pic)
            im = np.array(im)
            if((im.shape[0] == 256) and (im.shape[1] == 256) and count < 90):
                if(count < 5):
                    x_test = np.concatenate((x_test, im.reshape((1,)+shape)), axis=0)
                    y_test = np.concatenate((y_test, [[label]]), axis=0)
                else:
                    x_train = np.concatenate((x_train, im.reshape((1,)+shape)), axis=0)
                    y_train = np.concatenate((y_train, [[label]]), axis=0)
                count = count + 1
        label = label + 1
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


def preprocess(x_train, x_test, y_train, y_test, factor, n_classes):

    x_train /= factor
    x_test /= factor

    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':

    path = 'UCMerced_LandUse/Images/'
    factor = 255
    n_classes = 21

    x_train, x_test, y_train, y_test = read_images(path)
    x_train, x_test, y_train, y_test = preprocess(x_train, x_test, y_train, y_test, factor, n_classes)

    np.savez_compressed("uc_merced", x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
