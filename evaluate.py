import numpy as np
import h5py
from keras.models import load_model

#Load test set, load model, and evaluate on test set


# Predict on new data
predictions = model.predict(X_test)
rounded = [round(x[0], 2) for x in predictions]
#print(rounded)

# Compare predicted vs. known results
#print(y_test.shape)
rounds = np.asarray(rounded)
comps = np.vstack((y_test,rounds))
#print(comps.shape)
#for i in range(comps.shape[1]):
    #print(comps[:,i])

# Build confusion matrix
threshold = float(input('Please input an evaluation threshold: '))
def create_confusion_matrix(results):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for item in range(results.shape[1]):
        actual = results[0,item]
        predicted = results[1,item]
        if actual == 1 and predicted >= threshold:
            tp += 1
        elif actual == 0 and predicted >= threshold:
            fp += 1
        elif actual == 1 and predicted < threshold:
            fn += 1
        elif actual == 0 and predicted < threshold:
            tn += 1

    cm = np.array(([tp, fp], [fn, tn]))
    return cm

def model_metrics(mat):
    tp = float(mat[0,0])
    fp = float(mat[0,1])
    fn = float(mat[1,0])
    tn = float(mat[1,1])

    recall = round(tp/(tp+fn), 3)
    precision = round(tp / (tp + fp), 3)
    specificity = round(tn / (fp + tn), 3)
    accuracy = round((tp + tn) / (tp + fp +fn + tn), 3)

    print("Recall = {}".format(recall))
    print("Precision = {}".format(precision))
    print("Specificity = {}".format(specificity))
    print("Accuracy = {}".format(accuracy))

con_mat = create_confusion_matrix(comps)
print("Confustion Matrix")
print(con_mat)
model_metrics(con_mat)
