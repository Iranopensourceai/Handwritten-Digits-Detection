from scipy.io import loadmat
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import os

class LoadDigits:
    def __init__(self, mat_path):
        self.mat_path = mat_path
        self.features = np.squeeze(self.load_dict()["Data"])
        self.labels = self.load_dict()["labels"]

    # Parsing .mat file into dictionary
    def load_dict(self):
        return loadmat(self.mat_path)

    # Method for resizing all arrays to a single shape
    @staticmethod
    def resize(array: np.array, desire_shape=(20, 20)):
        final_array = np.zeros(shape=(array.shape[0], desire_shape[0]*desire_shape[1]))
        for i, arr in enumerate(array):
            if arr.shape[0] < desire_shape[0] and arr.shape[1] < desire_shape[1]:
                pad_needed = (desire_shape[0]-arr.shape[0])//2
                arr = np.pad(arr, pad_needed, constant_values=0, mode='constant')
                height_factor = desire_shape[0] / arr.shape[0]
                width_factor = desire_shape[1] / arr.shape[1]

            elif arr.shape[1]/arr.shape[0] > 0.3:
                height_factor = desire_shape[0]/arr.shape[0]
                width_factor = desire_shape[1]/arr.shape[1]

            else:
                pad_needed = (desire_shape[0] - arr.shape[1])//2
                arr = np.pad(arr, pad_needed, constant_values=0, mode='constant')
                width_factor = desire_shape[1] / arr.shape[1]
                height_factor = desire_shape[0]/arr.shape[0]

            mat = ndimage.zoom(arr, (height_factor, width_factor))
            mat[mat > 80] = 255
            mat[mat <= 80] = 0
            final_array[i] = mat.reshape(400)
        return final_array

    # Splitting data to train set and test set
    def train_test_split(self, features, labels, train_size=0.8):
        features = features//255
        labels = labels
        train_index = int(features.shape[0] * train_size)
        X_train, X_test = features[:train_index], features[train_index:]
        y_train, y_test = labels[:train_index], labels[train_index:]
        return X_train, X_test, y_train.ravel(), y_test.ravel()

    # Plot 10 images from each 10 digit classes
    @staticmethod
    def plot_digits(arr, label):
        fig, ax = plt.subplots(10, 10, figsize=(20, 20))
        for i, row in enumerate(ax):
            label_index = np.argwhere(label == i)
            for j, column in enumerate(row):
                column.imshow(arr.reshape(-1, 20, 20)[label_index][j][0], cmap='gray')
        plt.show()

    # Saving features and labels in .npy format
    @staticmethod
    def save(filename, file):
        np.save(os.path.join('dataset', f'{filename}.npy'), file)


DATASET_PATH = "dataset/Data_hoda_full.mat"
load_digits = LoadDigits(DATASET_PATH)
features, labels = load_digits.features, load_digits.labels
features_array = load_digits.resize(features)
# load_digits.plot_digits(features_array, labels)
X_train, X_test, y_train, y_test = load_digits.train_test_split(features_array, labels)
load_digits.save('X_train', X_train)
load_digits.save('X_test', X_test)
load_digits.save('y_train', y_train)
load_digits.save('y_test', y_test)

