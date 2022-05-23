"""
main module to launch the optical tweezers curve analysis and classification tool
"""
import os
import sys
from time import time
from datetime import date, datetime
from time import time
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from otanalysis.view.class_view import View
from otanalysis.controller.controller import Controller


def main():
    """
    Launch application for curve analyis
    """
    # my_os = sys.platform

    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    view = View()
    controller = Controller(view)
    view.set_controller(controller)
    view.show()
    app.exec()


START_TIME = time()
PATH_FILE = Path('data_test' + os.sep + 'txt' + os.sep)
PATH_FILE_JPK = Path("data_test" + os.sep + 'jpk_nt_force' + os.sep)
main()
print("--- %s seconds ---" % (time() - START_TIME))
