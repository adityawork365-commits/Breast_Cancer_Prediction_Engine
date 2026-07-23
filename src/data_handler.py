import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

def load_data():
    """
    Loads the Breast Cancer Wisconsin dataset from sklearn.
    Returns the raw features (X) and labels (y).
    """
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name='label')
    return X, y

def get_train_test_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets using stratified splitting.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
