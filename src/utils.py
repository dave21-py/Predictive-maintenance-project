import os
import logging
import yaml

def load_config(config_path="config/config.yaml"):
    """
    Reads the settings from the YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found in {config_path}")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def setup_logger(name, log_file):
    """
    Sets up the log file so we can see what the code is doing.
    Saves logs to 'logs/pipeline.log' AND prints to the screen.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # AVOID DUPLICATE LOGS
    if logger.handlers:
        return logger

    # File Handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.DEBUG)

    # Stream Handler (For Terminal)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)

    # Format logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(formatter)
    c_handler.setFormatter(formatter)

    logger.addHandler(f_handler)
    logger.addHandler(c_handler)

    return logger
