import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)    
    x_train, x_test = np.expand_dims(x_train, 3), np.expand_dims(x_test, 3)
    y_test_int = y_test # keep labels in integer form
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    return x_train, x_test, y_train, y_test, y_test_int


def sample_mnist_balanced(x_test, y_test_int, nsamples):
    '''
    Extract a random sample of nsamples elements from each category in test set
    '''
    samples = np.zeros((nsamples*10,28,28,1))
    labels = np.zeros((nsamples*10,))
    labels_int = []
    for i in range(10):
        labels_int.append([i]*nsamples)
        mask = y_test_int == i
        temp_x_test = x_test[mask]
        idx = np.random.permutation(temp_x_test.shape[0])[:nsamples]
        samples[i*nsamples: (i+1)*nsamples,:,:,:] = temp_x_test[idx]
        labels[i*nsamples: (i+1)*nsamples] = np.zeros((nsamples,)) + i
    labels = to_categorical(labels)
    return samples, labels
    