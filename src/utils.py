import random
import numpy as np
import torch
import json
import os
from datetime import datetime


def set_seed(seed=42):
    """Sets the seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_experiment(exp_name, config, results, report=None, folder="experiments"):
    """
    Saves experiment metadata, results, and classification reports.
    """
    # Create the timestamped experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(folder, f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Save config and metrics as JSON
    log_data = {
        "experiment_name": exp_name,
        "timestamp": timestamp,
        "config": config,
        "results": results
    }
    with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
        json.dump(log_data, f, indent=4)

    # 2. Save the human-readable classification report
    if report:
        with open(os.path.join(exp_dir, "classification_report.txt"), "w") as f:
            f.write(f"Experiment: {exp_name}\nDate: {timestamp}\n")
            f.write("=" * 40 + "\n")
            f.write(report)

    print(f"Results logged to: {exp_dir}")
    return exp_dir