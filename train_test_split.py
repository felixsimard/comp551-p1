'''
Custom train, test, split method used in notebooks.
'''

import numpy as np
import pandas as pd

def train_valid_split_CV(df, label_columns: list, frac=1, k=5):
    """
    Generator for k-fold cross validation.
    
    :param df: the dataframe containing the training and validation data
    :param label_columns: columns corresponding to the labels
    :param frac: fraction of the dataframe which should be used
    :param k: number of split in the cross validation
    :return: a tuple containing the training and validation data
    """
    # shuffle keep only a fraction of the entire dataframe    
    df = df.sample(frac=frac)
    split = np.array_split(df, k)
    for i in range(k):
        training_data = pd.concat([split[j] for j in range(k) if j != i], axis=0, sort=False)
        X_train = np.array(training_data.drop(label_columns, axis=1))
        Y_train = np.array(training_data[label_columns])
        
        validation_data = split[i]
        X_valid = np.array(validation_data.drop(label_columns, axis=1))
        Y_valid = np.array(validation_data[label_columns])
        
        yield X_train, X_valid, Y_train, Y_valid
