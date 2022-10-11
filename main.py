import matplotlib
from BuildTree import build_tree
from DataframeCreation import *
import pandas as pd
from Prediction import predict
from sklearn.model_selection import train_test_split
from Score import score
from Plot import draw_plt
matplotlib.use('TkAgg')


def main():
    data = pd.read_csv('agaricus-lepiota.data', sep=',')
    # task = create_dataframe(4, data, [1, 2, 3, 4], 8000)
    task = create_dataframe_randomly(4, data)
    attribute_values = task.iloc[:, 1:]
    class_values = task.iloc[:, 0]

    attribute_train, attribute_test, class_train, class_test = \
        train_test_split(attribute_values, class_values, train_size=0.7, random_state=36)

    tree = build_tree(task)
    class_predict = predict(attribute_test, tree)
    score(class_predict, class_test)
    print(tree)
    draw_plt(class_predict, class_test)


if __name__ == '__main__':
    main()
