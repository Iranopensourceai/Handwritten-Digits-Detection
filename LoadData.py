from scipy.io import loadmat
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
# import cv2


class LoadDigits:
    def __init__(self, train_mat_path, test_mat_path):
        self.train_mat_path = train_mat_path
        self.test_mat_path = test_mat_path
        self.train_features = np.squeeze(self.load_dict(self.train_mat_path)["Data"])
        self.train_labels = self.load_dict(self.train_mat_path)["labels"]
        self.test_features = np.squeeze(self.load_dict(self.test_mat_path)["Data"])
        self.test_labels = self.load_dict(self.test_mat_path)["labels"]

    # Parsing .mat file into dictionary
    def load_dict(self, mat_path):
        return loadmat(mat_path)

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
                pad_needed = abs((desire_shape[0] - arr.shape[1])//2)
                arr = np.pad(arr, pad_needed, constant_values=0, mode='constant')
                width_factor = desire_shape[1] / arr.shape[1]
                height_factor = desire_shape[0]/arr.shape[0]

            mat = ndimage.zoom(arr, (height_factor, width_factor))
            mat[mat > 80] = 255
            mat[mat <= 80] = 0
            final_array[i] = mat.reshape(desire_shape[0]*desire_shape[1])
        return final_array

    @staticmethod
    def skimage_resize(images, desire_shape=(20, 20)):
        array = np.zeros((images.shape[0], desire_shape[0]*desire_shape[1]))
        for i, img in enumerate(images):
            img = resize(img, desire_shape)
            array[i] = img.reshape(100)
        return array

    # Splitting data to train set and test set
    def train_test_split(self, features, labels, train_size=0.8):
        features = features/255
        labels = labels
        train_index = int(features.shape[0] * train_size)
        X_train, X_test = features[:train_index], features[train_index:]
        y_train, y_test = labels[:train_index], labels[train_index:]
        return X_train, X_test, y_train.ravel(), y_test.ravel()

    # Plot 10 images from each 10 digit classes
    @staticmethod
    def plot_digits(arr, label, images_shape=(10, 10)):
        fig, ax = plt.subplots(10, 10, figsize=(20, 20))
        for i, row in enumerate(ax):
            label_index = np.argwhere(label == i)
            for j, column in enumerate(row):
                column.imshow(arr.reshape(-1, images_shape[0], images_shape[1])[label_index][j][0], cmap='gray')
        plt.show()

    # Saving features and labels in .npy format
    @staticmethod
    def save(filename, file):
        np.save(os.path.join('dataset', f'{filename}.npy'), file)


TRAIN_DATASET_PATH = "dataset/Data_hoda_full.mat"
TEST_DATASET_PATH = "dataset/Test_20000.Mat"
load_digits = LoadDigits(TRAIN_DATASET_PATH, TEST_DATASET_PATH)
X_train, y_train = load_digits.train_features, load_digits.train_labels
X_test, y_test = load_digits.test_features, load_digits.test_labels

# Also you can use load_digits.resize() method instead. Try both of them and compare the results
X_train = load_digits.resize(X_train)
X_test = load_digits.resize(X_test)

load_digits.plot_digits(X_train, y_train)
load_digits.plot_digits(X_test, y_test)

# In case there is a need for validation data
# X_train, X_val, y_train, y_val = load_digits.train_test_split(X_train, y_train)

# Saving arrays (if needed)
# load_digits.save('X_train', X_train)
# load_digits.save('X_test', X_test)
# load_digits.save('X_val', X_val)
# load_digits.save('y_val', y_val)
# load_digits.save('y_train', y_train)
# load_digits.save('y_test', y_test)
