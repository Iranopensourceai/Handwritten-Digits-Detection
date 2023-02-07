import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle

image_train = np.load('data/X_train.npy')
label_train = np.load('data/y_train.npy')
image_test = np.load('data/X_test.npy')
label_test = np.load('data/y_test.npy')
image_val = np.load('data/X_val.npy')
label_val = np.load('data/y_val.npy')


def dataset_info():
    print('------------------Train data------------------')
    print('X_train (images) shape: ', image_train.shape)
    print('y_train (labels) shape: ', label_train.shape)
    print('------------------Test data-------------------')
    print('X_test (images) shape: ', image_test.shape)
    print('y_test (labels) shape: ', label_test.shape)
    print('---------------validation data----------------')
    print('X_validation (images) shape: ', image_val.shape)
    print('y_validation (labels) shape: ', label_val.shape)
    print('----------------------------------------------')


def my_plot(x, y, x_str):
    fig = plt.figure(figsize=(12, 4))
    for i in range(1, 5):
        fig.add_subplot(1, 4, i)
        plt.tight_layout()
        plt.title(x_str + '['+str(i)+'] = ' + str(y[i]))
        plt.imshow(x[i], cmap='gray')
    plt.show()


def visualize_samples_of_dataset():
    my_plot(image_train, label_train, 'image_train')
    my_plot(image_test, label_test, 'image_test')
    my_plot(image_val, label_val, 'image_val')


def train_model(N=3):
    model = KNeighborsClassifier(n_neighbors=N)
    model.fit(image_train, label_train)
    return model


dataset_info()
visualize_samples_of_dataset()
model = train_model()
with open('classify_knn', 'wb') as files:
    pickle.dump(model, files)
