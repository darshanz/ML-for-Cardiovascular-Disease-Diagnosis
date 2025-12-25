import ast
import json
import logging
import os

import numpy as np
from joblib import load, dump
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
import pandas as pd
from xgboost import XGBClassifier
from pathlib import Path


class Classifier():
    def __init__(self, config, project_name):
        self.project_name = project_name
        self.model_config = config

    def pred_sklearn_clf(self, model, X_train, y_train, X_test, y_test):
        """Train and predict using sklearn-like models."""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        return y_test, y_pred, y_prob, model

    def pred_xgb(self, X_train, y_train,  X_test, y_test, xgb_params): 
         
        model = XGBClassifier(**xgb_params) 
        model.fit(X_train.values, y_train.values) 
        preds = model.predict_proba(X_test.values)
        if xgb_params['num_classes'] > 2:
            y_pred = preds.argmax(axis=1) 
        else: 
            preds_binary_prob = preds[:, 1]
            y_pred = (preds_binary_prob > 0.5).astype(int)
            preds = preds_binary_prob 
        y_test_clean = y_test.values if hasattr(y_test, 'values') else y_test 
        return y_test_clean, y_pred, preds, model



    def _initialize_model(self, model_type, hyperparams): 
        if model_type == "logistic_regression":
            return LogisticRegression(**hyperparams)
        elif model_type == "random_forest":
            return RandomForestClassifier(**hyperparams)
        elif model_type == "decision_tree":
            return DecisionTreeClassifier(**hyperparams)
        elif model_type == "knn":
            return KNeighborsClassifier(**hyperparams)
        elif model_type == "svc":
            return SVC(probability=True, **hyperparams)
        elif model_type == "naive_bayes":
            return GaussianNB(**hyperparams)
        elif model_type == "adaboost":
            return AdaBoostClassifier(**hyperparams)
        elif model_type == "mlp":
            return MLPClassifier(**hyperparams)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def load_data(self):
        current_dir = Path.cwd()
        work_dir = current_dir.parent.parent
        data_dir = f'{work_dir}/data/{self.project_name}'
        X_train = load(f"{data_dir}/X_train.pkl")
        X_test = load(f"{data_dir}/X_test.pkl")
        y_train = load(f"{data_dir}/y_train.pkl")
        y_test = load(f"{data_dir}/y_test.pkl")
        return X_train, X_test, y_train, y_test

    def run(self): 
        X_train, X_test, y_train, y_test = self.load_data()

        model_results = {}
        name = self.model_config["name"]
        model_type = self.model_config["type"]
        params = self.model_config.get("hyperparameters", {})

        save_path = os.path.join("output", self.project_name, "results", name)
        os.makedirs(save_path, exist_ok=True)


        logging.info(f"Training model: {name}")  
        if model_type == "xgboost": 
            y_true, y_pred, y_prob, model = self.pred_xgb(X_train, y_train, X_test, y_test, params)
        else:
            model = self._initialize_model(model_type, params)
            y_true, y_pred, y_prob, model = self.pred_sklearn_clf(model, X_train, y_train, X_test, y_test)

        dump(y_true, f"{save_path}/y_true.pkl")
        dump(y_pred, f"{save_path}/y_pred.pkl")
        dump(y_prob, f"{save_path}/y_prob.pkl")

        specificity_binary  = 0 # Currently Binary only
        cm = confusion_matrix(y_test, y_pred)
        dump(cm, f"{save_path}/confusion_matrix.pkl")
        if cm.shape == (2, 2):  # binary
            tn, fp, fn, tp = cm.ravel()
            specificity_binary = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        self.metrics = {
            "f1": f1_score(y_test, y_pred, average='weighted'),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "specificity": specificity_binary,
        }
        if y_prob is not None:
            self.metrics["auroc"] = roc_auc_score(y_test, y_prob)
        else:
            self.metrics["auroc"] = np.nan

 
        dump(model, f"{save_path}/model.pkl") 
        dump(self.metrics, f"{save_path}/metrics.pkl")

        logging.info(f"Model {name} done") 
        self.model = model
        return self.metrics
 
