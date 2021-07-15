# Author: Ioannis Matzakos

from datetime import datetime


# Getting the current timestamp (date and time) of the system at the given moment, displays it and returns it
def get_timestamp():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return timestamp
