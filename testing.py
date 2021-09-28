from itertools import product
from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from train_test_split import train_valid_split_CV

# Define our target prediction columns for this dataset (whether a kind of mushroom is edible or poisonous)
#target_columns = ['x0_EDIBLE', 'x0_POISONOUS']
  

def test(model_cls, hyperparams_grid: Dict[str, list], ohe_df, target_columns):

    GROWING_SAMPLE_SIZES = [0.2, 0.4] #, 0.5, 0.75, 1.] 
    print(GROWING_SAMPLE_SIZES)
    TEST_FRACTION = 0.2 

    # Perform our custom cross validation
    # We use the outer loop to test our accuracies with growing sample sizes as aksed 
    # in the 'Experiments' section in the handout

    # Note: we assume that regardless of the total sample size taken, the test set represents 20% of the dataset and 
    # the train/validation set 80%.

    # General metrics
    mean_squared_errors_arr = []
    accuracies_per_sample_size = []

    for frac in GROWING_SAMPLE_SIZES:

        ohe_df_fraction = ohe_df.sample(frac=frac)
        train_valid_rows = int((1 - TEST_FRACTION) * len(ohe_df_fraction))
        train_valid_df = ohe_df_fraction.iloc[:train_valid_rows]
        test_df = ohe_df_fraction.iloc[train_valid_rows:]

        X_test = np.array(test_df.drop(target_columns, axis=1))
        y_test = np.array(test_df[target_columns])

        # Plot accuracies for each K
        best_accuracies_per_k = {}

        print("Using", frac*100, "% of total dataset.\n") 

        # Keep track of metrics to determine best k value
        training_validation_metric = {}

        # Loop over K values
        for hyperparams_values in product(*hyperparams_grid.values()):
            current_hyper_params = dict(zip(hyperparams_grid.keys(), hyperparams_values))
            print("------- current hp =", current_hyper_params, "-------")
            
            # metrics for cross validation
            accuracies_cv = []
            scores_cv = []
            temp_dfs_cv = []

            # Perform cross-validation for a certain K
            for X_train, X_valid, y_train, y_valid in train_valid_split_CV(train_valid_df, target_columns, 1, 5):
            
                # Fit model using current K value and current train-valid-test sets
                clf = model_cls(**current_hyper_params)
                clf.fit(X_train, y_train)

                # Predict on validation set
                y_pred = clf.predict(X_valid)
                acc_score_cv = accuracy_score(y_pred, y_valid)

                # Append result to our list of accuracies
                accuracies_cv.append(acc_score_cv)
                scores_cv.append(clf.score(X_train, y_train))
                temp_dfs_cv.append([X_train, y_train])
            
            # Report mean of training and validation metrics for this K value
            mean_accuracy_cv = np.mean(accuracies_cv)
            mean_score_cv = np.mean(scores_cv)

            # Append metric
            training_validation_metric[hyperparams_values] = (mean_accuracy_cv, mean_score_cv)

        import pprint
        pprint.pprint(training_validation_metric)
        # PLOT K VALUES ACCURACIES

#         plt.figure()
#         hyp = list(hyperparams_grid.keys())[0]
#         x = list(hyperparams_grid[hyp])
#         valid_scores = list(v[0] for v in training_validation_metric.values())
#         training_scores = list(v[1] for v in training_validation_metric.values())
#         plt.plot(x, training_scores, label='training score')
#         plt.plot(x, valid_scores, label='valid score')
#         plt.xticks(np.arange(min(x), max(x) + 1, 1))
#         plt.legend()
#         plt.show()

        # Determine best K-value for this fraction size
        # best_accuracy = max()
        # best_accuracies.append(best_accuracy)
        # best_accuracy_index = accuracies.index(best_accuracy)
        # best_split_df = temp_dfs[best_accuracy_index]

        mean_metrics = {values: np.mean([m1, m2]) for values, (m1, m2) in training_validation_metric.items()}
        values, v = max(*mean_metrics.items(), key=lambda item: item[1])
        
        # Save optimal K
        best_hyper_params = dict(zip(hyperparams_grid.keys(), values))
        print('Best hps', best_hyper_params)

        # Re-train model with optimal K value and full training-validation set
        clf = model_cls(**best_hyper_params)
        X_trainfull = np.array(train_valid_df.drop(target_columns, axis=1))
        y_trainfull = np.array(train_valid_df[target_columns])
        clf.fit(X_trainfull, y_trainfull)

        # Predict on test set
        predicted = clf.predict(X_test)

        # Analysis
        test_accuracy_score = accuracy_score(predicted, y_test)
        training_score = clf.score(X_trainfull, y_trainfull)
        training_error = mean_squared_error(y_test, predicted)
        mean_squared_errors_arr.append(training_error*100)

        # Analyze/Show/Plot metrics
        print("Mean Squared Error:", training_error)
        print("Accuracy on training set:", training_score)
        print("Accuracy on test set:", test_accuracy_score)
        # print("Mean:", np.mean(accuracies))
        # print("Standard Deviation:", np.std(accuracies))
        # print("Variance:", np.var(accuracies))
        # print("\n")

        # Show a confusion matrix
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
        cm = confusion_matrix(y_test.argmax(axis=1), predicted.argmax(axis=1))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()
        plt.show()

        print("-----------------------------")

    # Plot mean model accuracy for each k-value used for a certain sample size
    # print("Accuracy of model for each K-value used")
    # k_plots = plt.figure(frac)
    # plt.plot(K_VALUES, best_accuracies)

    print("\n\n")
    print('Mean squared errors', mean_squared_errors_arr)

    # Plot mean squared errors as fraction of dataset considered grows
    # print(GROWING_SAMPLE_SIZES.shape, mean_squared_errors_arr.shape)
    # plt.plot(GROWING_SAMPLE_SIZES, mean_squared_errors_arr)
    # plt.show()

#model_cls = KNeighborsClassifier
#hyperparams_grid = {'n_neighbors': [1,2,3,4,5,6,7], 'weights': ['uniform', 'distance']}
#target_columns = ['x0_EDIBLE', 'x0_POISONOUS']
#test(model_cls, hyperparams_grid, ohe_df, target_columns)
