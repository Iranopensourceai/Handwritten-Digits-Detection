import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle

X_train = np.load('data/X_train.npy')
label_train = np.load('data/y_train.npy')


def dataset_info():
    print('------------------Train data------------------')
    print('X_train (images) shape: ', X_train.shape)
    print('y_train (labels) shape: ', label_train.shape)
    print('----------------------------------------------')


def visualize_samples_of_dataset():
    fig = plt.figure(figsize=(12, 4))
    for i in range(1, 5):
        fig.add_subplot(1, 4, i)
        plt.tight_layout()
        plt.title('X_train['+str(i)+'] = ' + str(label_train[i]))
        plt.imshow(X_train[i], cmap='gray')
    plt.show()


def train_model(N=3):
    dataset_info()
    visualize_samples_of_dataset()
    model = KNeighborsClassifier(n_neighbors=N)
    model.fit(X_train, label_train)
    return model


model = train_model()
with open('model_pkl', 'wb') as files:
    pickle.dump(model, files)
