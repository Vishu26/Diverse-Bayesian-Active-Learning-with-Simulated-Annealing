from data import get_data
from model import LeNet
from query import ALSA, ClusterMargin, BADGE, BALD, CoreSet, DBAL, Random, ClusterSampling, UALSA, RandomPoolALSA
import numpy as np

data = 'mnist'
method = 'ALSA'

if data=='mnist':
    X_train, y_train, X_test, y_test = get_data('mnist')
    input_shape = (28, 28, 1)
    n_classes = 10
    initial_size = 32

elif data=='fashion_mnist':
    X_train, y_train, X_test, y_test = get_data('fashion_mnist')
    input_shape = (28, 28, 1)
    n_classes = 10
    initial_size = 32
    
elif data=='cifar':
    X_train, y_train, X_test, y_test = get_data('cifar10')
    input_shape = (32, 32, 1)
    n_classes = 10
    initial_size = 1000

else:
    raise "Invalid Dataset"

active_learner = LeNet(input_shape)
labeled_idx, size, acc = [], [], []

if method != 'ALSA' and method != 'UALSA' and method !='RandomPoolALSA':
    idx = np.random.choice(range(len(X_train)), size=initial_size).tolist()
    hist = active_learner.fit(X_train[idx], y_train[idx], epochs=50, batch_size=128, validation_data=(X_test, y_test),verbose=0)
    score = max(hist.history['val_accuracy'])
    size.append(len(idx))
    acc.append(score)
    labeled_idx.extend(idx)
    print(F"Initial Accuracy {score}")

if method == 'ALSA':
    ## Hyperparameters ##
    alpha = 200
    beta, e, m, k, p = 5, 0.1, 10, 32, 1
    epochs = 600
    query = ALSA(X_train, alpha, beta, e, m, k, p, epochs)
elif method == 'ClusterMargin':
    ## Hyperparameters ##
    k, p, eps = 32, 5000, 3
    epochs = 20
    query = ClusterMargin(active_learner, X_train, k, p, eps)
elif method == 'BADGE':
    ## Hyperparameters ##
    k, p = 32, 5000
    epochs = 20
    query = BADGE(k, p)
elif method == 'BALD':
    ## Hyperparameters ##
    k = 32
    epochs = 20
    query = BALD(k)
elif method == 'CoreSet':
    ## Hyperparameters ##
    k, p = 32, 5000
    epochs = 20
    query = CoreSet(k, p)
elif method == 'DBAL':
    ## Hyperparameters ##
    k = 32
    epochs = 20
    query = DBAL(k)
elif method == 'Random':
    ## Hyperparameters ##
    k = 32
    epochs = 20
    query = Random(k)
elif method == 'ClusterSampling':
    ## Hyperparameters ##
    m, k, p = 10, 4, 100
    epochs = 20
    query = ClusterSampling(X_train, m, k, p)
elif method == 'UALSA':
    ## Hyperparameters ##
    alpha = 200
    beta, e, m, k, p = 5, 0.1, 10, 32, 1
    epochs = 600
    query = UALSA(X_train, alpha, beta, e, m, k, p, epochs)
elif method == 'RandomPoolALSA':
    ## Hyperparameters ##
    alpha = 200
    beta, e, m, k, p = 5, 0.1, 10, 32, 32
    epochs = 600
    query = RandomPoolALSA(X_train, alpha, beta, e, m, k, p, epochs)
else:
    raise "Invalid Method"

epoch=0
while epoch < epochs-1:
    
    index, epoch = query.query(active_learner, X_train, labeled_idx, epoch)
    labeled_idx.extend(index)
    hist = active_learner.fit(X_train[labeled_idx], y_train[labeled_idx], epochs=50, batch_size=128, validation_data=(X_test, y_test),verbose=0)
    score = max(hist.history['val_accuracy'])
    size.append(len(labeled_idx))
    acc.append(score)
    print(F"Accuracy: {score}, No of Annotations: {size[-1]}")

np.save(F"{method}_{data}.npy", list(zip(size, acc)))