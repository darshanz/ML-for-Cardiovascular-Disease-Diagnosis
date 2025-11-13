import pandas as pd
from pathlib import Path
import logging
import os

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "expt.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger("shap").disabled = True
    logging.getLogger().addHandler(logging.StreamHandler())  # also print to console
    return log_dir