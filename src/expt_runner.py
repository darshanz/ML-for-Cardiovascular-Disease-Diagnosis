import json
import logging
import os
from classifier import Classifier
from explainer import Explainer
import mlflow



class ExperimentRunner:
    def __init__(self, dataset, config, tracking_uri="sqlite:///cardiovascular.db"):
        self.config = config
        self.models = config['models']
        self.project_name = dataset
 
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config['experiment_name'])

        mlflow.sklearn.autolog(log_models=False)
        mlflow.xgboost.autolog(log_models=False)
 

    def run(self):
        logging.info("Experiment started")  
        model_results = {}
        for model_cfg in self.models:

            model_name = model_cfg["name"]
            run_name = f"{self.project_name}__{model_name}"

            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("dataset", self.project_name)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("model_type", model_cfg.get("type"))

                # Log hyperparameters
                hps = model_cfg.get("hyperparameters", {})
                for k, v in hps.items():
                    mlflow.log_param(f"hp.{k}", v)

                classifier_step = Classifier(model_cfg, project_name=self.project_name)
                logging.info(f"Starting  classifier for {model_cfg['name']}") 
                clf_results = classifier_step.run()
                logging.info(f"classifier step for {model_cfg['name']} completed: results={clf_results} ")
                mlflow.log_metrics({k: float(v) for k, v in clf_results.items()})

                if model_cfg.get("type") == "xgboost":
                    classifier_step.model._estimator_type = "classifier"
                    mlflow.xgboost.log_model(classifier_step.model,  name=f"{model_cfg['name']}_model")
                else:
                    mlflow.sklearn.log_model(classifier_step.model,  name=f"{model_cfg['name']}_model")


                artifacts_dir = os.path.join("output", self.project_name, "results", model_name)
                if os.path.isdir(artifacts_dir):
                    mlflow.log_artifacts(artifacts_dir, artifact_path="outputs")

                model_results[model_cfg["name"]] = {
                "results": clf_results 
                }

            
            explainer = Explainer(model_name=model_cfg['name'], project_name=self.project_name) 
            logging.info(f"Starting  Explainer step for {model_cfg['name']}")
            explainer.run() # Save plots
            logging.info(f"Explainer step for {model_cfg['name']} completed")

      
        # Save Results
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", f"{self.project_name}_results.json")

        with open(output_path, "w") as f:
            json.dump(model_results, f, indent=4)
        logging.info("Experiments finished successfully")
