import os

import numpy as np
import pandas as pd
from joblib import load
from matplotlib import pyplot as plt 

import builtins
import shap
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

class Explainer():
    def __init__(self,  model_name, project_name): 
        self.model_name = model_name
        self.project_name = project_name

    def run(self):
        current_dir = Path.cwd()
        work_dir = current_dir.parent.parent
        data_dir = f'{work_dir}/data/{self.project_name}'
        X_test = load(f"{data_dir}/X_test.pkl")
        X_train = load(f"{data_dir}/X_train.pkl")
        results_dir = os.path.join('output', self.project_name, "results", self.model_name)

        loaded_model = load(f"{results_dir}/model.pkl")

        def model_predict(data):
            if hasattr(loaded_model, 'predict_proba'):
                return loaded_model.predict_proba(data)[:, 1]
            else:
                return loaded_model.predict(data)

        X_background = shap.sample(X_train, 100)
        explainer = shap.KernelExplainer(model_predict, X_background)
        shap_values = explainer.shap_values(X_test)
          
 
        save_path = os.path.join("output", self.project_name, "xai", self.model_name)
        os.makedirs(save_path, exist_ok=True)

        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(f'{save_path}/shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.clf()

        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig(f'{save_path}/shap_summary_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.clf()

        for i, column_name in enumerate(X_test.columns):
            shap.dependence_plot(i, shap_values, X_test, show=False)
            plt.savefig(f'{save_path}/shap_dependence_plot_{column_name}.png', dpi=300, bbox_inches='tight')
            plt.clf()