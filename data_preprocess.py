import numpy as np
from PIL import Image
import os
import glob
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


def read_images(path='UCMerced_LandUse/Images/', shape=(64, 64, 3)):
    xtr, ytr, xt, yt = [], [], [], [] 
    dirs = os.listdir(path)
    label = 0
    for i in tqdm(dirs):
        count = 0
        imgs = glob.glob(path+i+'/*.jpg')
        idx = np.split(imgs, [3*len(imgs)//4])
        for pic in idx[0]:
            im = Image.open(pic)
            im = np.array(im)
            if((im.shape[0] == shape[0]) and (im.shape[1] == shape[1])):
                xtr.append(im)
                ytr.append([label])
        for pic in idx[1]:
            im = Image.open(pic)
            im = np.array(im)
            if((im.shape[0] == shape[0]) and (im.shape[1] == shape[1])):
                xt.append(im)
                yt.append([label])
        label = label + 1
    return np.array(xtr).astype(np.float32), np.array(xt).astype(np.float32), np.array(ytr).astype(np.uint8), np.array(yt).astype(np.uint8)


def preprocess(x_train, x_test, y_train, y_test, factor, n_classes):

    x_train /= factor
    x_test /= factor

    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':

    path = 'EuroSat/2750/'
    factor = 255
    n_classes = 10
    img_shape = (64, 64, 3)

    x_train, x_test, y_train, y_test = read_images(path)
    x_train, x_test, y_train, y_test = preprocess(x_train, x_test, y_train, y_test, factor, n_classes)

    np.savez_compressed("eurosat", x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
