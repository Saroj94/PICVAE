import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

##dynamically find the project root, assuming that the logger is inside the src folder
BASE_DIR = Path(__file__).resolve().parent.parent

##Constants for logs
LOG_DIR = "logs"

##Use YYYY-MM-DD so files sorts chronological folders
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
MAX_LOG_SIZE = 5*1024*1024
BACKUP_COUNT = 3

##Construct paths using pathlib
log_dir_path = BASE_DIR/LOG_DIR ##joins the root folder (BASE_DIR) and the logs folder name ("logs") together as folders
log_dir_path.mkdir(parents=True, exist_ok=True) ##create the log folder, infact the parent folder is missing then create parent folder as well
log_file_path=log_dir_path/LOG_FILE ##create the log file with (.log) at the end of the file

def configure_logger():
    """
    Configures logging with file handler and console handler.
    """

    ##HOW TO WRITE THE LOG FILE
    ##create a custom logger 
    logger = logging.getLogger("picvae_ivim_project")
    logger.setLevel(logging.DEBUG) #python built-in logging levels

    # Prevent duplicate handlers if this function runs multiple times
    if logger.hasHandlers():
        logger.handlers.clear()


    # Define the line number to the formatter and added filename
    formatter = logging.Formatter(
        "[%(asctime)s] %(lineno)d %(filename)s - %(levelname)s - %(message)s"
    )

    ##create handler that decides where the log messages go
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    ## HOW TO PRINT LOG IN TERMINAL/CONSOLE
    console_handler = logging.StreamHandler() ##strem handler because of continuous logging
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO) ##INFO: only produce information of steps, no thousands of scrolling text

    ##add both handlers: file_handler and console_handler into logger file
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

##configure and expose the logger objects

logger = configure_logger()  
