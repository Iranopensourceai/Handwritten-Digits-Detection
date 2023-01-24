from scipy.io import loadmat
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize


class LoadDigits:
    def __init__(self, train_mat_path, test_mat_path, remaining_mat_path):
        self.train_mat_path = train_mat_path
        self.test_mat_path = test_mat_path
        self.remaining_mat_path = remaining_mat_path

    # Parsing .mat file into dictionary
    @staticmethod
    def load_dict(mat_path):
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
        return final_array/255

    @staticmethod
    def skimage_resize(images, desire_shape=(20, 20)):
        array = np.zeros((images.shape[0], desire_shape[0]*desire_shape[1]))
        for i, img in enumerate(images):
            img = resize(img, desire_shape)
            array[i] = img.reshape((desire_shape[0]*desire_shape[1]))
        return array/255

    # Splitting data to train set and test set
    def read_dataset(self, val='remaining'):
        train_images = np.squeeze(self.load_dict(self.train_mat_path)["Data"])
        train_labels = self.load_dict(self.train_mat_path)["labels"]
        if val == 'remaining':
            test_images = np.squeeze(self.load_dict(self.test_mat_path)["Data"])
            test_labels = self.load_dict(self.test_mat_path)["labels"]
            val_images = np.squeeze(self.load_dict(self.remaining_mat_path)["Data"])
            val_labels = self.load_dict(self.remaining_mat_path)["labels"]
        elif val == 'test':
            test_images = np.squeeze(self.load_dict(self.remaining_mat_path)["Data"])
            test_labels = self.load_dict(self.remaining_mat_path)["labels"]
            val_images = np.squeeze(self.load_dict(self.test_mat_path)["Data"])
            val_labels = self.load_dict(self.test_mat_path)["labels"]
        else:
            raise ValueError("Invalid name for val argument use 'remaining' or 'test'")
        return train_images, train_labels, test_images, test_labels.ravel(), val_images.ravel(), val_labels.ravel()

    # Plot 10 images from each 10 digit classes
    @staticmethod
    def plot_digits(arr, label, images_shape=(20, 20)):
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
VAL_DATASET_PATH = "dataset/Remainingset_Hoda.Mat"
load_digits = LoadDigits(TRAIN_DATASET_PATH, TEST_DATASET_PATH, VAL_DATASET_PATH)

# pass 'remaining' or 'test' to choose your validation dataset
img_train, label_train, img_test, label_test, img_val, label_val = load_digits.read_dataset(val='remaining')

# Also you can use load_digits.resize() method instead. Try both of them and compare the results
img_train = load_digits.skimage_resize(img_train)
img_test = load_digits.skimage_resize(img_test)
img_val = load_digits.skimage_resize(img_val)

load_digits.plot_digits(img_train, label_train)
load_digits.plot_digits(img_test, label_test)
load_digits.plot_digits(img_val, label_val)


# Saving arrays (if needed)
load_digits.save('X_train', img_train)
load_digits.save('X_test', img_test)
load_digits.save('X_val', img_val)
load_digits.save('y_train', label_train)
load_digits.save('y_test', label_test)
load_digits.save('y_val', label_val)
