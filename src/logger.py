import logging
from .config import Config
import os

def setup_logger():
    config = Config()
    log_file = os.path.join(os.path.dirname(__file__), '..', config.logging['log_file'])
    log_level = getattr(logging, config.logging['log_level'].upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('WBTech_Project')
    return logger
