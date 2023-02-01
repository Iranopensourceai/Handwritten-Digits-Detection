import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle


def train_model(N=3):
    X_train = np.load('X_train')
    label_train = np.load('y_train')

    model = KNeighborsClassifier(n_neighbors=N)
    model.fit(X_train, label_train)
    return model


model = train_model()
with open('model_pkl', 'wb') as files:
    pickle.dump(model, files)
