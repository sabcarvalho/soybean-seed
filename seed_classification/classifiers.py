"""Sklearn-based classifiers (SVM, RF, XGBoost)."""
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from .config import CLASS_LABELS


class Classifier:
    def __init__(self):
        self.model = None

    def eval(self, x_train, y_train, x_test, y_test):
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        print("\n--- RESULTADO---")
        print(classification_report(y_test, y_pred))
        self.plot_confusion_matrix(y_test, y_pred)

    def cross_val(self, dataset, label, cv_folds=5, random_state=42):
        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        scores = cross_val_score(
            self.model, dataset, label, cv=cv,
            error_score="raise", scoring="accuracy",
        )
        print(f"Scores per fold: {scores}")
        print(f"Mean Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_LABELS)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()


class SupportVectorMachine(Classifier):
    def __init__(self, random_state=42):
        self.model = SVC(kernel="linear", random_state=random_state)


class RandomForest(Classifier):
    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(n_estimators=300, random_state=random_state)


class XGBoost(Classifier):
    def __init__(self, n_classes=4, random_state=42):
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=random_state,
            tree_method="hist",
            n_estimators=80,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        )