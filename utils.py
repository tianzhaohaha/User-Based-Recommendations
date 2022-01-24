import numpy as np
import pandas as pd
import math


def read_train():
    data = pd.read_csv("./datasets/train_set.csv")
    data.drop('timestamp', axis=1, inplace=True)
    return data

def read_test():
    test = pd.read_csv('datasets/test_set.csv')
    test.drop('timestamp', axis=1, inplace=True)
    return test

def pearson(X,Y):
    """

    :param X:
    :param Y:
    :return: cos(x,y)
    """
    up = 0
    down_x = 0
    down_y = 0
    for i in range(len(X)):
        up += X[i] * Y[i]
        down_x += X[i] * X[i]
        down_y += Y[i] * Y[i]
    return up / (math.sqrt(down_x) * math.sqrt(down_y))


def movie_to_id(df):
    movie_to_id = {}
    j = 0
    for i in range(len(df)):
        if df["movieId"][i] not in movie_to_id:
            movie_to_id[df["movieId"][i]] = j
            j += 1
    return movie_to_id


