from os import sep
import logging
from time import time
from datetime import date, datetime
from pathlib import Path


def create_logger():
    today = str(date.today())
    time_today = str(datetime.now().time().replace(
        microsecond=0)).replace(':', '-')
    path_log = Path("Log")
    path_log.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        filename=path_log.__str__() + sep + "otanalysis" + today +
                        '_' + time_today + ".log",
                        filemode="a",
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('logger_otanalysis')
    
