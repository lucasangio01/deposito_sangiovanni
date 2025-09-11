import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


def load_data(path):
    data = pd.read_csv(path)
    return data

def split_data(data):
    X = data.drop("Class", axis = 1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test

def make_predictions_tree(X_train, y_train, X_test):
    tree = DecisionTreeClassifier(max_depth=5, class_weight="balanced" ,random_state=42)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    return y_pred

def make_predictions_rf(X_train, y_train, X_test):
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return y_pred

def oversample(X_train, y_train):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled
