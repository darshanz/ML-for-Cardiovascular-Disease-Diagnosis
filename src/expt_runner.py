import json
import logging
import os
from classifier import Classifier
from explainer import Explainer


class ExperimentRunner:
    def __init__(self, dataset, config):
        self.config = config
        self.models = config['models']
        self.project_name = dataset
 

    def run(self):
        logging.info("Experiment started")  
        model_results = {}
        for model_cfg in self.models:
            classifier_step = Classifier(model_cfg, project_name=self.project_name)
            explainer = Explainer(model_name=model_cfg['name'], project_name=self.project_name)

            # Compute stage scores
            logging.info(f"Starting  classifier for {model_cfg['name']}") 
            clf_results = classifier_step.run()
            logging.info(f"classifier step for {model_cfg['name']} completed: results={clf_results} ")
 
            logging.info(f"Starting  Explainer step for {model_cfg['name']}")
            explainer.run() # Save plots
            logging.info(f"Explainer step for {model_cfg['name']} completed")

            model_results[model_cfg["name"]] = {
                "results": clf_results 
            }
 

        # Save Results
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", "results.json")

        with open(output_path, "w") as f:
            json.dump(model_results, f, indent=4)
        logging.info("Experiments finished successfully")
