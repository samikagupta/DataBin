import copy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import pandas as pd
from sklearn.neural_network import MLPClassifier
import gen_bins
from sklearn.tree import RandomForestClassifier
from sklearn.metrics import accuracy_score

def preprocess(data, bin_name, y_name, bin_col, random_state=42):
    """
    Preprocess data, train a classifier, and evaluate its performance.

    This function assigns a specified binning column to the dataset, splits the data
    into training and testing sets using a custom bin-aware splitting method, trains
    a classification model, and computes the test accuracy. It is primarily used as
    a preprocessing and evaluation step for downstream error analysis (e.g., MRE).

    Args:
        data (pandas.DataFrame): The input dataset containing features and target variable.
        bin_name (str): The name of the column to be added/overwritten in `data` to store bin assignments.
        y_name (str): The name of the target (label) column.
        bin_col (array-like): The bin assignments corresponding to each row in `data`.
        random_state (int, optional): Seed for reproducibility in data splitting and model training.
            Defaults to 42.

    Returns:
        
        clf: Trained classification model (currently RandomForestClassifier).
        test_x (pandas.DataFrame): Test feature set.
        test_label (pandas.Series or array-like): True labels for the test set.
            - acc (float): Accuracy score of the model on the test set.

    Notes:
        Data splitting is performed using `gen_bins.train_test`, which may incorporate
          bin-aware logic.
        Alternative classifiers (e.g., MLP, XGBoost) can be used by uncommenting the
          corresponding lines in the function.
        The function changes data by adding/replacing the bin_name column.
    """
    data[bin_name] = bin_col
    y = data[y_name]

    # is used for XGB model
    # data[y_name] = data[y_name].astype('category').cat.codes

    # split the data using either the sklearn function or use the default setting already done, 
    # see function in gen_bins 
    train_x, train_label, test_x, test_label =  gen_bins.train_test(data, data.columns, y_name, random_state)
    # train_x, test_x, train_label, test_label = train_test_split(data, y, train_size=0.2, random_state=random_state)

    # chose which classification model to use, just uncommnent 
    clf = RandomForestClassifier(random_state=random_state)
    # clf = MLPClassifier(random_state=random_state, max_iter=500)
    # clf = xgb.XGBClassifier(n_estimators=100, max_depth=3,enable_categorical=True, learning_rate=0.1)
    
    # fit model and get accuracy score 
    clf.fit(train_x, train_label)
    acc = accuracy_score(test_label, clf.predict(test_x))

    # return all relevant attributes in order to calc MRE 
    return clf, test_x, test_label, acc

def get_bin_model(data, y_name, random_state=42):
    """
    Use a weak learner model to train data. This is the most
    simplest version of this version (no sampling version), just take the model

    Args:
        data: pd.DataFrame
        y_name: str, Target class of data
        random_state: int, should be randomly generated

    Returns:
        Trained classifier (clf) and test_train_split data
    """
    data2 = copy.deepcopy(data)
    y = data2[y_name]
    X = data2.drop(columns=[y_name]) 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    # other options for classification
    # clf = LogisticRegression().fit(X_train, y_train)
    clf = DecisionTreeClassifier(
        random_state=random_state, criterion="gini",
    )

    clf.fit(X_train, y_train)
    return clf, X_train, X_test, y_train, y_test




