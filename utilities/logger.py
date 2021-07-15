# Author: Ioannis Matzakos | Date: 22/07/2019

import logging
import sys

from utilities.date_utils import get_timestamp


# Configure the logging system
def setup_logger(name):
    # setup logging format
    formatter = logging.Formatter(fmt='%(asctime)s | %(levelname)s | %(module)s : %(message)s')
    # display log in console
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    # save log in a file
    timestamp = get_timestamp()
    filehandler = logging.FileHandler(f"logs/detect_covid_19_in_xray_images_{timestamp}.log")
    filehandler.setFormatter(formatter)
    # setup logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(filehandler)
    return logger
