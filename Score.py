from sklearn.metrics import recall_score, accuracy_score, precision_score


def score(class_predict, class_test):
    print("Recall ", recall_score(class_test, class_predict, average="micro"))
    print("Accuracy ", accuracy_score(class_test, class_predict))
    print("Precision ", precision_score(class_test, class_predict, average="micro"))
