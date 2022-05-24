#!/usr/bin/python
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout


class Infowindow(QWidget):
    """
    Instantiation of info window
    """
    def __init__(self, parent=None):
        """
        Secondary window constructor
        """
        super(Infowindow, self).__init__(parent)
        self.title = QLabel(" Loading Data...")
        self.nb_curve = QLabel("0/6")
        self.info_curve = QLabel("")
        self.setWindowTitle(u'Processing data')
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.title)
        self.setLayout(self.main_layout)

    #################################################################

    def set_nb_curve(self, nb):
        """
        setter of the 'nb_curve' widget text
        """
        self.nb_curve.setText(nb)
        self.main_layout.addWidget(self.nb_curve)
        self.show()

    ##################################################################

    def set_info_curve(self, text):
        """
        setter of the 'info_curve' widget text
        """
        self.info_curve.setText(text)
        self.main_layout.addWidget(self.info_curve)
        self.show()
    
    ###################################################################
    
    def set_title(self):
        """
        setter of the 'title' widget text
        """
        self.title.setText('Loading is done')
