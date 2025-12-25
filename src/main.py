import yaml 
from utils import setup_logger
from expt_runner import ExperimentRunner 

    

if __name__ == "__main__":
    datasets = ['cleveland', 'hungarian', 'switzerland', 'longbeach_va']
    
    for dataset in datasets: 
        with open(f"config.yaml") as f:
            config = yaml.safe_load(f)
        log_dir = setup_logger("logs")
        expt_runner = ExperimentRunner(dataset, config)
        expt_runner.run()

    

    