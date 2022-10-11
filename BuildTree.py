import numpy as np
import pandas as pd


def get_subtable(df: pd.DataFrame, node, value):
    ndf = df[df[node] == value].reset_index(drop=True)
    ndf.drop(node, inplace=True, axis=1)
    return ndf


def find_entropy(df: pd.DataFrame):
    class_name = df.keys()[0]
    labels = df[class_name].tolist()
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * np.log2(norm_counts)).sum()


def find_entropy_attribute(df: pd.DataFrame, key: str):
    key_labels = df[key].tolist()
    key_values, key_counts = np.unique(key_labels, return_counts=True)
    norm_key_counts = key_counts / key_counts.sum()
    key_entropies = list(map(lambda value: find_entropy(get_subtable(df, key, value)), key_values))
    return (norm_key_counts * key_entropies).sum()


def split_info(df: pd.DataFrame, feature: str):
    feature_labels = df[feature].tolist()
    feature_values, feature_counts = np.unique(feature_labels, return_counts=True)
    norm_feature_counts = feature_counts / feature_counts.sum()
    return -(norm_feature_counts * np.log2(norm_feature_counts)).sum()


def find_winner_node(df: pd.DataFrame):
    gain_ratio = []
    for key in df.keys()[1:]:
        split = split_info(df, key)
        if split == 0:
            gain_ratio.append(np.inf)
        else:
            gain_ratio.append(find_entropy(df) - find_entropy_attribute(df, key) / split)
    return df.keys()[1:][np.argmax(gain_ratio)]


def build_tree(df: pd.DataFrame, tree=None):
    if tree is None:
        tree = {}
    class_name = df.keys()[0]
    node = find_winner_node(df)
    tree[node] = {}
    node_values = np.unique(df[node])

    for value in node_values:
        sub_df = get_subtable(df, node, value)
        cl_values, cl_counts = np.unique(sub_df[class_name], return_counts=True)

        if len(cl_counts) == 1:
            tree[node][value] = cl_values[0]
        elif len(sub_df.columns) == 1:
            return cl_values[np.argmax(cl_counts)]
        else:
            tree[node][value] = build_tree(sub_df)

    return tree
