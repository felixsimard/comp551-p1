from train_test_split import train_valid_split_CV

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from typing import List
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay

GROWING_SAMPLE_SIZES = [0.05, 0.8] # [0.2, 0.4, 0.8, 1.0]
TEST_FRACTION = 0.2
K_VALUES = [k for k in range(3, 5)]

def iter_test(ohe_df: pd.DataFrame, model_type: str)->List[float]:
    
    for frac in GROWING_SAMPLE_SIZES:
      ohe_df_fraction = ohe_df.sample(frac=frac)
      train_valid_rows = int((1 - TEST_FRACTION) * len(ohe_df_fraction))
      train_valid_df = ohe_df_fraction.iloc[:train_valid_rows]
      test_df = ohe_df_fraction.iloc[train_valid_rows:]

      X_test = np.array(test_df.drop(target_columns, axis=1))
      y_test = np.array(test_df[target_columns])

      best_accuracies = []

      print("Using", frac*100, "% of total dataset.\n") 