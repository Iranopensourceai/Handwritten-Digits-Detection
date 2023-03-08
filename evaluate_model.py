import LoadData
import pickle
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

with open('model', 'rb') as files:
    pickle.load(model, files)


def predict_model(model, data):
    y_pred = model.predict(data)
    return y_pred


y_pred_test = predict_model(model, TEST_DATASET_PATH)
y_pred_val = predict_model(model, VAL_DATASET_PATH)


def evaluate_model(y_test, y_pred):
    cl_report = metrics.classification_report(y_test, y_pred)

    return {'cl_report': cl_report}


def Confusion_matrix(model, y_test, y_pred):
    cm = metrics.confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()


# Print result
# Evaluate Model
result_test = evaluate_model(y_test, y_pred_test)
print('classification_report', result_test['cl_report'])

result_pred = evaluate_model(y_test, y_pred_val)
print('classification_report', result_pred['cl_report'])

Confusion_matrix(model, y_test, y_pred_test)
Confusion_matrix(model, y_test, y_pred_val)
