import matplotlib.pyplot as plt


# create a function to evaluate the model
def evaluation_model(x_test, y_test):
    result = model.evaluate(x_test, y_test)
    print(model.metrics_names)
    return result


# display accuracy and val accuracy

def plotting_accuracy_metrics(acc, val_acc):
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    return plt.show()


# display loss and val loss

def plotting_loss_metrics(loss, val_loss):
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    return plt.show()


evaluation_model(X_test, y_test)
plotting_accuracy_metrics(history.history['accuracy'], history.history['val_accuracy'])
plotting_accuracy_metrics(history.history['loss'], history.history['val_loss'])
