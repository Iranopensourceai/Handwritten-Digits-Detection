def evaluate_model(model, y_test, y_pred):
    from sklearn import metrics

    # Calculate accuracy, precision, recall, f1-score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, average="weighted")
    rec = metrics.recall_score(y_test, y_pred, average="weighted")
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")

    # Display confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'cm': cm}


# Print result
# Evaluate Model
result = evaluate_model(model, y_test, y_pred)

print('Accuracy:', result['acc'])
print('Precision:', result['prec'])
print('Recall:', result['rec'])
print('F1 Score:', result['f1'])
print('Confusion Matrix:\n', result['cm'])
