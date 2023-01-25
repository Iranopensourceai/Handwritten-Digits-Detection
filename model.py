import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from .LoadData import LoadDigits


def train_model(X_train, y_train, X_validation, y_validation, N=3):
    models = []
    scores = []
    sub_validation_length = int(len(X_validation) / 3)
    sub_validations = [[0, sub_validation_length], [
        sub_validation_length, 2*sub_validation_length], [2*sub_validation_length, -1]]
    for i, sub_validation in enumerate(sub_validations):
        new_train_fold = X_train.copy()
        new_label_fold = y_train.reshape(y_train.shape[0]).copy()
        for j in range(len(sub_validation)):
            if not i == j:
                new_train_fold = np.append(
                    new_train_fold, X_validation[sub_validations[j][0]:sub_validations[j][1]], axis=0)
                new_label_fold = np.append(
                    new_label_fold, y_validation[sub_validations[j][0]:sub_validations[j][1]], axis=0)
        model = KNeighborsClassifier(n_neighbors=N)
        model.fit(X_train, y_train)
        y_pred = model.predict(
            X_validation[sub_validations[i][0]:sub_validations[i][1]])
        y_test = y_validation[sub_validations[i][0]:sub_validations[i][1]]
        # Calculate accuracy
        acc = metrics.accuracy_score(y_test, y_pred)
        models.append(model)
        scores.append(acc)
    return models[scores.index(max(scores))]


TRAIN_DATASET_PATH = "dataset/Data_hoda_full.mat"
TEST_DATASET_PATH = "dataset/Test_20000.Mat"
VAL_DATASET_PATH = "dataset/Remainingset_Hoda.Mat"
load_digits = LoadDigits(
    TRAIN_DATASET_PATH, TEST_DATASET_PATH, VAL_DATASET_PATH)

img_train, label_train, img_test, label_test, img_val, label_val = load_digits.read_dataset(
    val='remaining')

# Also you can use load_digits.resize() method instead. Try both of them and compare the results
img_train = load_digits.skimage_resize(img_train)
img_test = load_digits.skimage_resize(img_test)
img_val = load_digits.skimage_resize(img_val)

model = train_model(img_train, label_train, img_val, label_val, 3)
