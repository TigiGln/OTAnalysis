#!/usr/bin/python
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QLabel, QWidget, QGridLayout


class GraphView(QWidget):
    """
    Instantiation of manual optical correction graph window
    """

    def __init__(self, parent=None):
        """
        Initialization of a graphic window
        """
        super(GraphView, self).__init__(parent)
        self.main_layout = QGridLayout()
        self.setLayout(self.main_layout)
