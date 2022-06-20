import torch
import torch.nn as nn
import numpy as np
from linear_cca import linear_cca
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from DeepCCAModels import DeepCCA
from utils import load_data, svm_classify
import time
import logging
try:
    import pickle as thepickle
except ImportError:
    import pickle as thepickle

import gzip
import numpy as np
import torch.nn as nn
import torch 
import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import cv2 as cv2
from matplotlib import units
import numpy as np
import matplotlib.pyplot as plt

def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import pickle as thepickle
    except ImportError:
        import pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret

def make_tensor(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = torch.tensor(data_x)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y

"""loads the data from the gzip pickled files, and converts to numpy arrays"""

print('loading data ...')
f = gzip.open('../noisymnist_view2.gz', 'rb')
train_set, valid_set, test_set = load_pickle(f)
f.close()

train_set_x, train_set_y = make_tensor(train_set)
valid_set_x, valid_set_y = make_tensor(valid_set)
test_set_x, test_set_y = make_tensor(test_set)

i = 0

temp = train_set_x[i].numpy()
temp2 = np.reshape(temp, (28,28), order='F')
temp2 = cv2.resize(temp2, (240,240))

fig, axes = plt.subplots(nrows = 2, ncols = 4)
temp = train_set_x[i].numpy()
temp2 = np.reshape(temp, (28,28), order='F')
temp2 = cv2.resize(temp2, (240,240))
axes[1,0].set_title('Noisy: 5')
axes[1,0].imshow(temp2, cmap = "gray")
temp = train_set_x[1].numpy()
temp2 = np.reshape(temp, (28,28), order='F')
temp2 = cv2.resize(temp2, (240,240))
axes[1,1].set_title('Noisy: 0')
axes[1,1].imshow(temp2, cmap = "gray")
temp = train_set_x[2].numpy()
temp2 = np.reshape(temp, (28,28), order='F')
temp2 = cv2.resize(temp2, (240,240))
axes[1,2].set_title('Noisy: 4')
axes[1,2].imshow(temp2, cmap = "gray")
temp = train_set_x[3].numpy()
temp2 = np.reshape(temp, (28,28), order='F')
temp2 = cv2.resize(temp2, (240,240))
axes[1,3].set_title('Noisy: 1')
axes[1,3].imshow(temp2, cmap = "gray")


f = gzip.open('../noisymnist_view1.gz', 'rb')
train_set, valid_set, test_set = load_pickle(f)
f.close()

train_set_x, train_set_y = make_tensor(train_set)
valid_set_x, valid_set_y = make_tensor(valid_set)
test_set_x, test_set_y = make_tensor(test_set)

temp = train_set_x[i].numpy()
temp2 = np.reshape(temp, (28,28), order='F')
temp2 = cv2.resize(temp2, (240,240))
axes[0,0].set_title('Normal: 5')
axes[0,0].imshow(temp2, cmap = "gray")
temp = train_set_x[1].numpy()
temp2 = np.reshape(temp, (28,28), order='F')
temp2 = cv2.resize(temp2, (240,240))
axes[0,1].set_title('Normal: 0')
axes[0,1].imshow(temp2, cmap = "gray")
temp = train_set_x[2].numpy()
temp2 = np.reshape(temp, (28,28), order='F')
temp2 = cv2.resize(temp2, (240,240))
axes[0,2].set_title('Normal: 4')
axes[0,2].imshow(temp2, cmap = "gray")
temp = train_set_x[3].numpy()
temp2 = np.reshape(temp, (28,28), order='F')
temp2 = cv2.resize(temp2, (240,240))
axes[0,3].set_title('Normal: 1')
axes[0,3].imshow(temp2, cmap = "gray")

plt.show()

