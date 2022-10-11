import pandas as pd


def solve(line: pd.DataFrame, tree: dict):

    if type(tree) is not dict:
        return tree

    tree_root = list(tree.keys())[0]
    sub_tree = tree[tree_root]
    root_values = list(sub_tree.keys())

    for value in root_values:
        line_value = line.iloc[0][tree_root]
        if line_value == value:
            return solve(line, sub_tree[value])


def predict(df: pd.DataFrame, tree: dict):
    result = []
    for i in range(len(df)):
        result.append(solve(df.iloc[[i]], tree))
    return result
