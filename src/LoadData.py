from scipy.io import loadmat
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import os


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
        """
        Code for resizing images 
        In case you need padding, you can uncomment following lines
        """
        final_array = np.zeros(shape=(array.shape[0], desire_shape[0]*desire_shape[1]))
        for i, arr in enumerate(array):
            height_factor = desire_shape[0] / arr.shape[0]
            width_factor = desire_shape[1] / arr.shape[0]
            # if arr.shape[0] < desire_shape[0] and arr.shape[1] < desire_shape[1]:
            #     pad_needed = (desire_shape[0]-arr.shape[0])//2
            #     arr = np.pad(arr, pad_needed, constant_values=0, mode='constant')
            #     height_factor = desire_shape[0] / arr.shape[0]
            #     width_factor = desire_shape[1] / arr.shape[1]

            # elif arr.shape[1]/arr.shape[0] > 0.3:
            #     height_factor = desire_shape[0]/arr.shape[0]
            #     width_factor = desire_shape[1]/arr.shape[1]

            # else:
            #     pad_needed = abs((desire_shape[0] - arr.shape[1])//2)
            #     arr = np.pad(arr, pad_needed, constant_values=0, mode='constant')
            #     width_factor = desire_shape[1] / arr.shape[1]
            #     height_factor = desire_shape[0]/arr.shape[0]
            mat = ndimage.zoom(arr, (height_factor, width_factor))
            mat[mat > 80] = 255
            mat[mat <= 80] = 0
            final_array[i] = mat.reshape(desire_shape[0]*desire_shape[1])
        return final_array/255


    # Splitting data to train, test and validation set
    def read_dataset(self, desire_shape):
        train_images = self.resize(np.squeeze(self.load_dict(self.train_mat_path)["Data"]),
                                           desire_shape=desire_shape)
        train_labels = np.squeeze(self.load_dict(self.train_mat_path)["labels"])
        test_images = self.resize(np.squeeze(self.load_dict(self.test_mat_path)["Data"]),
                                            desire_shape=desire_shape)
        test_labels = np.squeeze(self.load_dict(self.test_mat_path)["labels"])
        val_images = self.resize(np.squeeze(self.load_dict(self.remaining_mat_path)["Data"]),
                                            desire_shape=desire_shape)
        val_labels = np.squeeze(self.load_dict(self.remaining_mat_path)["labels"])
        return train_images, train_labels, test_images, test_labels, val_images, val_labels

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
        np.save(os.path.join('data', f'{filename}.npy'), file)


TRAIN_DATASET_PATH = "data/Trainset_Hoda.mat"
TEST_DATASET_PATH = "data/Testset_Hoda.Mat"
VAL_DATASET_PATH = "data/Remainingset_Hoda.Mat"
load_digits = LoadDigits(TRAIN_DATASET_PATH, TEST_DATASET_PATH, VAL_DATASET_PATH)

shape = (20, 20)
img_train, label_train, img_test, label_test, img_val, label_val = load_digits.read_dataset(desire_shape=shape)

# load_digits.plot_digits(img_train, label_train, images_shape=shape)
# load_digits.plot_digits(img_test, label_test, images_shape=shape)
# load_digits.plot_digits(img_val, label_val, images_shape=shape)


# Saving arrays (if needed)
load_digits.save('X_train', img_train)
load_digits.save('X_test', img_test)
load_digits.save('X_val', img_val)
load_digits.save('y_train', label_train)
load_digits.save('y_test', label_test)
load_digits.save('y_val', label_val)
