import pandas as pd
import random


def create_dataframe_randomly(n: int, original: pd.DataFrame):
    df = pd.DataFrame()
    df['class'] = original['class'].tolist()
    for position in range(1, n + 1):
        random_index = random.randint(1, len(original.columns))
        random_column_name = original.columns[random_index]
        random_column_as_list = original[random_column_name].tolist()
        df[random_column_name] = random_column_as_list
    return df


def create_dataframe(n: int, original: pd.DataFrame, indices_list: list, columns_amount: int):
    df = pd.DataFrame()
    df['class'] = original['class'].tolist()
    for position in range(1, n + 1):
        index = indices_list[position - 1]
        column_name = original.columns[index]
        column_as_list = original[column_name].tolist()
        df[column_name] = column_as_list
    return df.iloc[:columns_amount]
