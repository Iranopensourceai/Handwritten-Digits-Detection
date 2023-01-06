import numpy as np
import scipy.io
from sklearn.neighbors import KNeighborsClassifier


mat = scipy.io.loadmat('dataset/Data_hoda_full.mat')

max_i = 0
max_j = 0
for x in mat['Data']:
    i, j = x[0].shape
    max_i = max(i, max_i)
    max_j = max(j, max_j)

X_train = []
for x in mat['Data']:
    X_train.append(np.resize(x[0], (max_i*max_j)))

y_train = mat['labels'].reshape(60000)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
