"""
Class View
"""
from time import sleep
from os import sep
from datetime import date, datetime
from pathlib import Path
from PyQt5.QtWidgets import QWidget, QFileDialog, QFrame, QSpinBox, QApplication, QProgressBar
from PyQt5.QtWidgets import QPushButton, QRadioButton, QHBoxLayout, QVBoxLayout, QLabel, QMessageBox
from PyQt5.QtWidgets import QLineEdit, QGridLayout, QGroupBox, QDoubleSpinBox, QButtonGroup
from PyQt5.QtWidgets import QScrollArea, QMainWindow
from PyQt5.QtCore import Qt, pyqtSignal, QEvent, QEventLoop, QTimer
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from view.class_info import Infowindow
from view.class_toggle import QtToggle
from view.class_graph_view import GraphView
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from re import match


class View(QMainWindow, QWidget):
    """
    Class instantiating windows
    """
    # pylint: disable=too-many-instance-attributes
    keyPressed = pyqtSignal(QEvent)

    def __init__(self):
        """
        Constructor of our 'View' class initializing the main grid of the window
        """
        QWidget.__init__(self)
        self.controller = None
        self.info = Infowindow()
        self.msgBox = QMessageBox()
        self.keyPressed.connect(self.on_key)
        self.nb_save_graph = 0
        self.setWindowIcon(QIcon('../pictures' + sep + 'icon.png'))
        self.info.setWindowIcon(QIcon('../pictures' + sep + 'icon.png'))
        self.setWindowTitle("View")
        self.size_window()

    ###############################################################################################################

    def size_window(self):
        """
        initialization parameters for widget management according to screen resolution
        """
        # screen = QApplication.desktop().screen
        self.screen_display = QApplication.desktop().screenGeometry()
        if self.screen_display.height() < 1000:
            self.scrollArea = QScrollArea()
            self.widget = QWidget()
            self.main_layout = QGridLayout()
            self.initialize_window()
            self.widget.setLayout(self.main_layout)
            self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.scrollArea.setWidgetResizable(True)
            self.setGeometry(0, 0, self.screen_display.width(),
                             self.screen_display.height())
            self.scrollArea.setGeometry(
                0, 0, self.screen_display.width(), self.screen_display.height()*2)
            self.scrollArea.setWidget(self.widget)
            self.setCentralWidget(self.scrollArea)
        else:
            self.widget = QWidget()
            self.main_layout = QGridLayout()
            self.initialize_window()
            self.widget.setLayout(self.main_layout)
            self.setCentralWidget(self.widget)
            print(self.widget.geometry())
            #self.main_layout.addWidget(self.toogle, 0, 1, 1, 1)

    ###############################################################################################################

    def initialize_window(self):
        """
        Enables the starting display with the addition of widgets to launch the analysis
        """
        self.n = 0
        self.check_supervised = True
        self.check_graph = False
        self.option = False
        self.check_global_local_graph = False
        self.abscissa_curve = False
        self.check_methods = False
        self.check_close_figure = False
        self.check_bilan = False
        self.count_select_plot = 0
        self.dict_fig_open = {}
        self.clear()
        self.data_description()
        self.create_button_select_data()
        self.create_model_radio()
        self.create_physical_parameters()
        self.create_alignement_incomplete_parameters()
        self.create_condition_parameters_type()
        self.create_select_correction_optical()

    ###########################################################################################################

    def change_button(self, button, name_button, method):
        """
        Allows you to change the method of a button
        """
        button.setText(name_button)
        button.disconnect()
        button.clicked.connect(method)

    #######################################################################################################

    def data_description(self):
        """
        Creation of widgets for the description of the analysis performed
        """
        layout_grid = QGridLayout()
        frame = QFrame()
        self.title_description = QLabel("Data Description")
        self.title_description.setStyleSheet("QLabel {font-weight: bold;}")
        self.condition = QLabel("condition 1")
        self.input_condition = QLineEdit()
        self.drug = QLabel("drug/condition 2")
        self.input_drug = QLineEdit()
        self.input_condition.setPlaceholderText('aCD3')
        layout_grid.addWidget(self.condition, 0, 0)
        layout_grid.addWidget(self.input_condition, 0, 1)
        layout_grid.addWidget(self.drug, 1, 0)
        layout_grid.addWidget(self.input_drug, 1, 1)
        frame.setLayout(layout_grid)
        frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        frame.setLineWidth(3)
        frame.setMidLineWidth(3)
        self.main_layout.addWidget(self.title_description, 0, 0, 1, 1)
        self.main_layout.addWidget(frame, 1, 0, 1, 2)

    ##############################################################################################################

    def create_button_select_data(self):
        """
        Creation of the button for the transmission of curve files
        """
        self.button_select_files = QPushButton("Select Files")
        self.button_select_directory = QPushButton('Select Folder')
        self.button_select_files.clicked.connect(
            lambda: self.download_files('files'))
        self.button_select_directory.clicked.connect(
            lambda: self.download_files('directory'))
        self.main_layout.addWidget(self.button_select_directory, 2, 0, 1, 1)
        self.main_layout.addWidget(self.button_select_files, 2, 1, 1, 1)

    ##############################################################################################################

    def create_model_radio(self):
        """
        Creation of widgets to select the model for the fit
        """
        self.groupbox = QGroupBox("Model for fitting contact: ")
        self.groupbox.setStyleSheet('QGroupBox {font-weight:bold;}')
        self.hbox = QHBoxLayout()
        self.linear = QRadioButton("Linear")
        self.linear.setChecked(True)
        self.linear.clicked.connect(self.click_model)
        self.sphere = QRadioButton("Sphere")
        self.sphere.clicked.connect(self.click_model)
        self.hbox.addWidget(self.linear)
        self.hbox.addWidget(self.sphere)
        self.groupbox.setLayout(self.hbox)
        self.main_layout.addWidget(self.groupbox, 3, 0, 1, 2)
    ##############################################################################################################

    def click_model(self):
        if self.n == 0:
            self.window_geometry = self.geometry()
        if self.linear.isChecked():
            if self.n != 0:
                self.frame_physical.hide()
                self.physical_parameters.hide()
                print(self.window_geometry.height())
                #self.resize(self.window_geometry.width(), self.window_geometry.height())
                self.setGeometry(self.window_geometry)
                print(self.geometry())
        if self.sphere.isChecked():
            self.n += 1
            if self.n == 1:
                self.main_layout.addWidget(self.frame_physical, 5, 0, 1, 2)
                self.main_layout.addWidget(self.physical_parameters, 4, 0)
            else:
                self.frame_physical.show()
                self.physical_parameters.show()

    ##############################################################################################################

    def create_physical_parameters(self):
        """
        Creation of the widget for the transmission of the eta parameter for the calculation 
        of the Young's modulus according to the object's comprehensibility 
        """
        self.physical_parameters = QLabel("Physical Parameters")
        self.physical_parameters.setStyleSheet("QLabel {font-weight: bold;}")
        hlayout = QHBoxLayout()
        layout_grid = QGridLayout()
        self.frame_physical = QFrame()
        self.eta = QLabel("η (eta)")
        self.input_eta = QDoubleSpinBox()
        self.radius = QLabel("bead radius (µm)")
        self.input_radius = QDoubleSpinBox()
        self.input_eta.setValue(0.5)
        self.input_radius.setValue(1.00)
        self.input_radius.setSingleStep(0.1)
        layout_grid.addWidget(self.eta, 0, 0)
        layout_grid.addWidget(self.input_eta, 0, 1)
        layout_grid.addWidget(self.radius, 1, 0)
        layout_grid.addWidget(self.input_radius, 1, 1)
        self.frame_physical.setLayout(layout_grid)
        self.frame_physical.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.frame_physical.setLineWidth(3)
        self.frame_physical.setMidLineWidth(3)

    #############################################################################################################

    def create_alignement_incomplete_parameters(self):
        """
        Creation of widgets for the management of the good alignment of the curves on an axis
        """
        self.align_parameters = QLabel("Alignement & Incomplete Parameters")
        self.align_parameters.setStyleSheet("QLabel {font-weight: bold;}")
        layout_grid = QGridLayout()
        frame = QFrame()
        self.epsilon = QLabel("Fmax epsilon (%)")
        self.input_epsilon = QSpinBox()
        self.pulling = QLabel("pulling length min (%)")
        self.input_pulling = QSpinBox()
        self.input_epsilon.setValue(30)
        self.input_pulling.setValue(50)
        layout_grid.addWidget(self.pulling, 0, 0)
        layout_grid.addWidget(self.input_pulling, 0, 1)
        layout_grid.addWidget(self.epsilon, 1, 0)
        layout_grid.addWidget(self.input_epsilon, 1, 1)
        frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        frame.setLineWidth(3)
        frame.setMidLineWidth(3)
        frame.setLayout(layout_grid)
        self.main_layout.addWidget(self.align_parameters, 6, 0, 1, 2)
        self.main_layout.addWidget(frame, 7, 0, 1, 2)

    #######################################################################################################

    def create_condition_parameters_type(self):
        """
        Creation of widgets to set up the classification with the different thresholds
        """
        self.condition_type = QLabel("Conditions for Classification")
        self.condition_type.setStyleSheet("QLabel {font-weight: bold;}")
        layout_grid = QGridLayout()
        frame = QFrame()
        self.jump_force = QLabel("NAD if jump is < (pN)")
        self.input_jump_force = QDoubleSpinBox()
        self.jump_position = QLabel("AD if position is < (nm)")
        self.input_jump_position = QSpinBox()
        self.nb_point_jump = QLabel("AD if slope is < (pts)")
        self.input_nb_points_jump = QSpinBox()
        self.factor = QLabel("Factor overcome noise (xSTD)")
        self.input_factor = QDoubleSpinBox()
        # self.factor_optical = QLabel("Factor optical effect (xSTD)")
        # self.input_factor_optical = QDoubleSpinBox()
        self.input_jump_force.setValue(5.0)
        self.input_jump_position.setMaximum(5000)
        self.input_jump_position.setValue(200)
        self.input_nb_points_jump.setMaximum(5000)
        self.input_nb_points_jump.setValue(200)
        self.input_factor.setValue(4.0)
        # self.input_factor_optical.setValue(1.0)
        layout_grid.addWidget(self.jump_force, 0, 0)
        layout_grid.addWidget(self.input_jump_force, 0, 1)
        layout_grid.addWidget(self.jump_position, 1, 0)
        layout_grid.addWidget(self.input_jump_position, 1, 1)
        layout_grid.addWidget(self.nb_point_jump, 2, 0)
        layout_grid.addWidget(self.input_nb_points_jump, 2, 1)
        layout_grid.addWidget(self.factor, 3, 0)
        layout_grid.addWidget(self.input_factor, 3, 1)
        # layout_grid.addWidget(self.factor_optical, 4, 0)
        # layout_grid.addWidget(self.input_factor_optical, 4, 1)
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        frame.setLineWidth(3)
        frame.setMidLineWidth(3)
        frame.setLayout(layout_grid)
        self.main_layout.addWidget(self.condition_type, 8, 0, 1, 2)
        self.main_layout.addWidget(frame, 9, 0, 1, 2)
    ##########################################################################################################

    def create_select_correction_optical(self):
        """
        Creation of a group of radio buttons to activate the automatic optical correction
        """
        self.groupbox_optical = QGroupBox("Automatic optical correction")
        self.groupbox_optical.setStyleSheet('QGroupBox {font-weight:bold;}')
        self.hbox = QHBoxLayout()
        self.none_correction = QRadioButton("None")
        self.none_correction.setChecked(True)
        self.none_correction.clicked.connect(self.click_model)
        self.automatic_correction = QRadioButton("Correction")
        self.automatic_correction.clicked.connect(self.click_model)
        self.hbox.addWidget(self.none_correction)
        self.hbox.addWidget(self.automatic_correction)
        self.groupbox_optical.setLayout(self.hbox)
        self.main_layout.addWidget(self.groupbox_optical, 10, 0, 1, 2)

    ##########################################################################################################

    def set_controller(self, controller):
        """
        setter of the controller attribute to pass a "controller" object
        """
        self.controller = controller

    #########################################################################################################

    def download_files(self, select):
        """
        recovery of a file to be analyzed
        """
        files = []
        directory = None
        self.controller.clear()
        print("Loading file")
        if select == 'files':
            files = QFileDialog.getOpenFileNames(self, u"", u"..",
                                                 u"JPK Files (*.jpk-nt-force) ;; Text files(*.txt)")
            files = files[0]
            self.controller.files = files
        elif select == 'directory':
            directory = QFileDialog.getExistingDirectory(
                self, "Open folder", "..", QFileDialog.ShowDirsOnly)
            path_directory = Path(directory)
            self.controller.set_list_files(path_directory)
        if files != [] or directory != None:
            self.button_select_directory.deleteLater()
            self.button_select_files.deleteLater()
            self.button_load = QPushButton('Load methods')
            self.button_load.clicked.connect(self.load_methods)
            self.main_layout.addWidget(self.button_load, 2, 0, 1, 2)
            self.button_launch_analyze = QPushButton('Launch analysis')
            self.button_launch_analyze.clicked.connect(self.launch_analyze)
            self.button_launch_analyze.setStyleSheet(
                "QPushButton { background-color: green; }")
            self.main_layout.addWidget(self.button_launch_analyze, 11, 0, 1, 2)

    ####################################################################################################

    def load_methods(self):
        files_methods = QFileDialog.getOpenFileName(self, u"", u"",
                                                    u"TSV Files (*.tsv)")
        files_methods = files_methods[0]
        if files_methods != '':
            methods_data = pd.read_csv(files_methods, sep='\t', header=0)
            self.input_condition.setText(methods_data['condition'][0])

            if isinstance(methods_data['drug'][0], str):
                self.input_drug.setText(methods_data['drug'][0])
            self.input_radius.setValue(methods_data['bead_radius'][0])

            for child in self.groupbox.children():
                if isinstance(child, QRadioButton):
                    if methods_data['model'][0] == child.text():
                        child.setChecked(True)
            for child in self.groupbox_optical.children():
                if isinstance(child, QRadioButton):
                    if methods_data['optical'][0] == child.text():
                        child.setChecked(True)
            self.input_eta.setValue(methods_data['eta'][0])
            self.input_pulling.setValue(methods_data['pulling_length'][0])
            self.input_epsilon.setValue(methods_data['threshold_align'][0])
            self.input_jump_force.setValue(methods_data['jump_force'][0])
            self.input_jump_position.setValue(methods_data['jump_distance'][0])
            self.input_nb_points_jump.setValue(methods_data['jump_point'][0])
            self.input_factor.setValue(methods_data['factor_noise'][0])
            self.button_load.deleteLater()
            self.check_methods = True

    ####################################################################################################

    def launch_analyze(self):
        """
        launch of the analysis on all the files that the user transmits
        if the file formats found are correct and complete then creation 
        of a curve object by valid file
        """
        length = str(len(self.controller.files))
        self.msgBox.setWindowTitle("Title")
        self.msgBox.setText("Loading...\n" + "0/" + length)
        self.msgBox.show()
        sleep(1.0)
        print("launch")
        self.methods = {}
        model = ""
        optical = ""
        threshold_align = self.input_epsilon.value()
        self.methods['threshold_align'] = threshold_align
        pulling_length = self.input_pulling.value()
        self.methods['pulling_length'] = pulling_length
        if self.linear.isChecked():
            model = self.linear.text()
        else:
            model = self.sphere.text()
        self.methods['model'] = model
        eta = self.input_eta.value()
        self.methods['eta'] = eta
        bead_ray = self.input_radius.value()
        self.methods['bead_radius'] = bead_ray
        jump_force = self.input_jump_force.value()
        self.methods['jump_force'] = jump_force
        jump_distance = self.input_jump_position.value()
        self.methods['jump_distance'] = jump_distance
        jump_point = self.input_nb_points_jump.value()
        self.methods['jump_point'] = jump_point
        tolerance = self.input_factor.value()
        self.methods['factor_noise'] = tolerance
        if self.none_correction.isChecked():
            optical = self.none_correction.text()
        else:
            optical = self.automatic_correction.text()
        self.methods['optical'] = optical
        drug = ""
        if self.input_drug.text() == "":
            drug = "NaN"
        else:
            drug = self.input_drug.text()
        self.methods['drug'] = drug
        condition = ""
        if self.input_condition.text() == "":
            condition = "NaN"
        else:
            condition = self.input_condition.text()
        self.methods['condition'] = condition
        self.controller.set_list_curve(self.methods)
        if len(self.controller.dict_curve) != 0:
            self.choices_option()
        else:
            self.create_button_select_data()
        self.button_launch_analyze.deleteLater()
        if not self.check_methods:
            self.button_load.deleteLater()
    ####################################################################################################

    def choices_option(self):
        if self.option == False:
            groupbox_option = QGroupBox("Analysis mode and Save")
            groupbox_option.setStyleSheet('QGroupBox {font-weight:bold;}')
            vbox_option = QVBoxLayout()
            layout_grid_option = QGridLayout()
            self.display_graphs = QRadioButton("Supervised")
            self.display_graphs.setChecked(True)
            self.save_table = QRadioButton("Unsupervised")
            self.save_table_graphs = QRadioButton("... with graphs")
            self.display_graphs.clicked.connect(self.choices_option)
            self.save_table.clicked.connect(self.choices_option)
            self.save_table_graphs.clicked.connect(self.choices_option)
            vbox_option.addWidget(self.display_graphs)
            vbox_option.addWidget(self.save_table)
            vbox_option.addWidget(self.save_table_graphs)
            groupbox_option.setLayout(vbox_option)
            self.button_option = QPushButton("Supervised")
            self.button_option.setAutoDefault(True)
            self.button_option.clicked.connect(self.show_graphic)
            self.button_option.setStyleSheet(
                "QPushButton { background-color: green; }")
            layout_grid_option.addWidget(groupbox_option, 0, 0, 3, 1)
            layout_grid_option.addWidget(self.button_option, 1, 1, 1, 1)
            frame_option = QFrame()
            frame_option.setLayout(layout_grid_option)
            self.main_layout.addWidget(frame_option, 2, 0, 1, 2)
            self.option = True
        else:
            self.button_option.disconnect()
            if self.display_graphs.isChecked():
                self.button_option.setText("Supervised")
                self.button_option.setStyleSheet(
                    "QPushButton { background-color: green; }")
                self.button_option.clicked.connect(self.show_graphic)
            elif self.save_table.isChecked():
                self.button_option.setText("Save table")
                self.button_option.setStyleSheet(
                    "QPushButton { background-color: red; }")
                self.button_option.clicked.connect(self.save)
            else:
                self.button_option.setText("Save with graphs")
                self.button_option.setStyleSheet(
                    "QPushButton { background-color: red; }")
                self.button_option.clicked.connect(self.save_and_save_graphs)

    ####################################################################################################
    def info_processing(self, nb, length):
        """
        creation of the information window on the progress of the curve analysis
        """
        num_curve = int(nb.split('/')[0])

        if num_curve < length:
            # self.PBEtatAvancement = QProgressBar()# Création de la bar de progression
            # self.PBEtatAvancement.setRange(0, 100)
            loop = QEventLoop()
            self.msgBox.show()
            self.msgBox.setText("Loading...\n" + str(nb))
            QTimer.singleShot(0, loop.quit)
            loop.exec_()
        else:
            self.msgBox.close()
            self.info_processing_done(nb)

    ###################################################################################################

    def info_processing_done(self, nb):
        """
        Counting the number of curve types found during the analysis  
        """
        nb_nad = nb_ad = nb_tui = nb_tuf = nb_re = nb_none = 0
        for curve in self.controller.dict_curve.values():
            type_curve = curve.features['automatic_type']
            if curve.features['automatic_type'] == 'NAD':
                nb_nad += 1
            elif curve.features['automatic_type'] == 'AD':
                nb_ad += 1
            elif curve.features['automatic_type'] == 'FTU':
                nb_tuf += 1
            elif curve.features['automatic_type'] == 'ITU':
                nb_tui += 1
            elif curve.features['automatic_type'] == 'RE' or curve.features['automatic_type'] == None:
                nb_re += 1
        label = "files processing: " + nb + "\n"
        label += "\nvalid curves for analysis:  " + \
            str(len(self.controller.dict_curve))
        label += '\nNAD:' + str(nb_nad) + ' AD:' + str(nb_ad) + ' FTU:' + str(nb_tuf) \
            + ' ITU:' + str(nb_tui) + ' RE:' + str(nb_re)
        self.info.set_title()
        self.info.set_info_curve(label)

    ###################################################################################################
    def select_plot(self, visualization_data, abscissa_data):
        self.fig = None
        self.current_curve = None
        if visualization_data:
            self.check_global_local_graph = True
            if self.count_select_plot != 0:
                self.animate_toggle.show()
            if abscissa_data:
                self.abscissa_curve = True
                self.fig, self.current_curve = self.controller.show_plot(
                    self.n, 'time')
            else:
                self.abscissa_curve = False
                self.fig, self.current_curve = self.controller.show_plot(
                    self.n, 'distance')
        else:
            self.check_global_local_graph = False
            self.abscissa_curve = False
            if self.count_select_plot > 0 and self.check_supervised:
                self.animate_toggle.hide()
                self.animate_toggle.setChecked(False)
            self.fig, self.current_curve = self.controller.global_plot(self.n)

        if self.fig is not None:
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.canvas.mpl_connect('button_press_event', self.mousePressEvent)
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.canvas.draw()
            self.main_layout.addWidget(self.toolbar, 0, 0, 1, 6)
            self.main_layout.addWidget(self.canvas, 1, 0, 7, 6)
        self.count_select_plot += 1
        self.setFocus()
    ###################################################################################################

    def show_graphic(self):
        self.clear()
        self.setMouseTracking(True)
        if self.screen_display.height() > 1000:
            self.setGeometry(0, 0, self.screen_display.width(
            ), self.screen_display.height()-self.screen_display.height()//3)
        self.check_graph = True
        self.select_plot(self.check_global_local_graph, self.abscissa_curve)
        self.length_list_curve = len(self.controller.dict_curve.values())
        if self.fig is not None:
            self.navigation_button()
            self.button_supervised = QPushButton()
            self.button_supervised.setStyleSheet(
                "QPushButton { height: 2em; background-color: green;}")
            if self.check_supervised:
                self.supervision_panel()
            else:
                self.button_supervised.setText('Open supervision Panel')
                self.button_supervised.clicked.connect(
                    lambda: self.option_supervised(True))
                self.main_layout.addWidget(self.button_supervised, 9, 0, 1, 6)

    ###################################################################################################

    def navigation_button(self):
        button_next = QPushButton('Next')
        button_next.setFocusPolicy(Qt.StrongFocus)
        button_next.clicked.connect(self.next_plot)
        button_previous = QPushButton('Previous')
        button_previous.clicked.connect(self.previous_plot)
        
        if self.length_list_curve > 1:
            if self.n > 0:
                # switch to the second page previous button always present
                if self.n <= (self.length_list_curve-1):
                    # all pages between 2 pages and second last
                    if self.check_supervised:
                        self.main_layout.addWidget(button_previous, 1, 6, 1, 1)
                        self.main_layout.addWidget(button_next, 1, 7, 1, 1)
                    else:
                        self.main_layout.addWidget(button_previous, 8, 0, 1, 3)
                        self.main_layout.addWidget(button_next, 8, 3, 1, 3)
                    if self.n == (self.length_list_curve-1):
                        button_next.setDisabled(True)
                        if self.check_supervised:
                            button_bilan = QPushButton('Bilan')
                            button_bilan.clicked.connect(self.show_bilan)
                            self.main_layout.addWidget(
                                button_bilan, 8, 6, 1, 2)
            else:
                # first page show_plot if nb page > 1
                if self.check_supervised:
                    self.main_layout.addWidget(button_next, 1, 6, 1, 2)
                else:
                    self.main_layout.addWidget(button_next, 8, 0, 1, 6)
        else:
            # page show plot if single page
            button_next.setDisabled(True)
            if not self.check_supervised:
                self.main_layout.addWidget(button_next, 8, 0, 1, 6)
            else:
                self.main_layout.addWidget(button_next, 1, 6, 1, 2) 
            

    ###################################################################################################

    def save_supervision_panel(self):
        button_save_graph = QPushButton("Save graphics")
        button_save_graph.clicked.connect(self.save_graph)
        button_save = QPushButton()
        if self.n < len(self.controller.dict_curve)-1:
            button_save.setText("Stop and Save")
        else:
            button_save.setText("Save")
        button_save.clicked.connect(self.save)
        button_save.setStyleSheet("QPushButton { background-color: red; }")
        self.main_layout.addWidget(button_save_graph, 2, 6, 1, 1)
        self.main_layout.addWidget(button_save, 2, 7, 1, 1)

    ###################################################################################################
    def supervision_panel(self):
        self.check_supervised = True
        self.button_supervised.setText('Close supervision Panel')
        self.button_supervised.clicked.connect(
            lambda: self.option_supervised(False))
        self.main_layout.addWidget(self.button_supervised, 0, 6, 1, 2)
        self.dict_supervised = {}
        self.frame_supervised = QFrame()
        self.grid_supervised = QGridLayout()
        self.save_supervision_panel()
        # label for the name of each curve
        label_curve = QLabel(self.current_curve.file.strip())
        if self.current_curve.features['automatic_AL']['AL'] == 'No':
            self.alignment_supervised()
        self.grid_supervised.addWidget(label_curve, 0, 0, 1, 1)
        self.toggle_all_curve_characteristics_point()
        self.toggle_abscissa_curve()
        self.valid_fit()
        self.toggle_optical_effect()
        self.type_supervised()
        self.frame_supervised.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.frame_supervised.setLineWidth(3)
        self.frame_supervised.setMidLineWidth(3)
        self.frame_supervised.setLayout(self.grid_supervised)
        self.dict_supervised[self.current_curve.file] = self.frame_supervised
        self.main_layout.addWidget(self.frame_supervised, 4, 6, 3, 2)
        label_num_curve = QLabel("     Courbe N°")
        self.num_curve_current = QLineEdit(str(self.n+1))
        self.num_curve_current.setMaximumWidth(50)
        self.num_curve_current.setAlignment(Qt.AlignRight)
        count_curves = QLabel('/' + str(self.length_list_curve))
        count_curves.setAlignment(Qt.AlignLeft)
        self.main_layout.addWidget(
            label_num_curve, 7, 6, 1, 1)
        self.main_layout.addWidget(
            self.num_curve_current, 7, 6, 1, 1, alignment=Qt.AlignRight)
        self.main_layout.addWidget(count_curves, 7, 7, 1, 1)

    ###################################################################################################
    def change(self):
        num_page = self.num_curve_current.text()
        regex = match("^[0-9]+$", num_page)
        if regex:
            num_page = int(num_page)
            if isinstance(num_page, int):
                if 0 < num_page <= self.length_list_curve:
                    self.n = num_page - 1
                    self.show_graphic()

    ###################################################################################################

    def alignment_supervised(self):
        axe_align = ""
        if len(self.current_curve.features['automatic_AL']['axe']) > 1:
            for index_axe in range(0, len(self.current_curve.features['automatic_AL']['axe']), 1):
                if index_axe < len(self.current_curve.features['automatic_AL']['axe'])-2:
                    axe_align += self.current_curve.features['automatic_AL']['axe'][index_axe] + ', '
                elif index_axe < len(self.current_curve.features['automatic_AL']['axe'])-1:
                    axe_align += self.current_curve.features['automatic_AL']['axe'][index_axe]
                else:
                    axe_align += ' and ' + \
                        self.current_curve.features['automatic_AL']['axe'][index_axe]
        else:
            axe_align = self.current_curve.features['automatic_AL']['axe'][0]
        label_alignment = QLabel("<html><img src= \".." + sep + "pictures" +
                                 sep + "warning.png\" /></html> Bad alignement on " + axe_align)
        group_box_align = QGroupBox('Accept')
        group_align = QButtonGroup()
        hbox_align = QHBoxLayout()
        yes_align = QRadioButton('Yes')
        no_align = QRadioButton('No')
        group_align.addButton(yes_align)
        group_align.addButton(no_align)
        hbox_align.addWidget(yes_align)
        hbox_align.addWidget(no_align)
        group_box_align.setLayout(hbox_align)
        for button in group_align.buttons():
            button.clicked.connect(lambda: self.modif_supervised('AL'))
            button.setFocusPolicy(Qt.NoFocus)
            if button.text() == str(self.current_curve.features['AL']):
                button.setChecked(True)
        self.grid_supervised.addWidget(label_alignment, 1, 0, 1, 1)
        self.grid_supervised.addWidget(group_box_align, 1, 1, 1, 1)

    ####################################################################################################
    def valid_fit(self):
        group_button_fit_press = QGroupBox("Press: Fit valid ?")
        group_fit_press = QButtonGroup()
        hbox_fit_press = QHBoxLayout()
        yes_press = QRadioButton('True')
        no_press = QRadioButton('False')
        if 'valid_fit_press' not in self.current_curve.features:
            no_press.setChecked(True)
        group_fit_press.addButton(yes_press)
        group_fit_press.addButton(no_press)
        hbox_fit_press.addWidget(yes_press)
        hbox_fit_press.addWidget(no_press)
        group_button_fit_press.setLayout(hbox_fit_press)
        for button in group_fit_press.buttons():
            button.clicked.connect(
                lambda: self.modif_supervised('valid_fit_press'))
            button.setFocusPolicy(Qt.NoFocus)
            if 'valid_fit_press' in self.current_curve.features:
                if button.text() == self.current_curve.features['valid_fit_press']:
                    button.setChecked(True)

        # radio group for the validation of the "Pull" segment fit
        group_button_fit_pull = QGroupBox("Pull: Fit valid ?")
        group_fit_pull = QButtonGroup()
        hbox_fit_pull = QHBoxLayout()
        yes_pull = QRadioButton('True')
        no_pull = QRadioButton('False')
        if 'valid_fit_pull' not in self.current_curve.features:
            no_pull.setChecked(True)
        group_fit_pull.addButton(yes_pull)
        group_fit_pull.addButton(no_pull)
        hbox_fit_pull.addWidget(yes_pull)
        hbox_fit_pull.addWidget(no_pull)
        group_button_fit_pull.setLayout(hbox_fit_pull)
        for button in group_fit_pull.buttons():
            button.clicked.connect(
                lambda: self.modif_supervised('valid_fit_pull'))
            button.setFocusPolicy(Qt.NoFocus)
            if 'valid_fit_pull' in self.current_curve.features:
                if button.text() == self.current_curve.features['valid_fit_pull']:
                    button.setChecked(True)
        self.grid_supervised.addWidget(group_button_fit_press, 2, 0, 1, 2)
        self.grid_supervised.addWidget(group_button_fit_pull, 4, 0, 1, 2)

    ##################################################################################################

    def toggle_all_curve_characteristics_point(self):
        self.toggle_display = QtToggle(
            120, 30, '#777', '#ffffff', '#0cc03c', 'Overview', 'Analyzed')
        self.toggle_display.setChecked(self.check_global_local_graph)
        self.toggle_display.clicked.connect(lambda: self.select_plot(
            self.toggle_display.isChecked(), self.animate_toggle.isChecked()))
        self.main_layout.addWidget(
            self.toggle_display, 3, 7, 1, 1, alignment=Qt.AlignRight)
    ##################################################################################################

    def toggle_abscissa_curve(self):
        self.animate_toggle = QtToggle(
            110, 30, '#777', '#ffffff', 'orange', 'Distance', 'Time')
        self.animate_toggle.setChecked(self.abscissa_curve)
        self.animate_toggle.clicked.connect(lambda: self.select_plot(
            self.toggle_display.isChecked(), self.animate_toggle.isChecked()))
        self.grid_supervised.addWidget(
            self.animate_toggle, 0, 1, 1, 1, alignment=Qt.AlignRight)
        if not self.check_global_local_graph:
            self.animate_toggle.hide()
    #################################################################################################

    def toggle_optical_effect(self):
        # Corrected optical effect
        grid_optical = QGridLayout()
        label_optical = QLabel("Correction optical effect")
        toggle_optical = QtToggle(
            120, 30, '#777', '#ffffff', '#4997d0', 'None', 'Manual')
        toggle_optical.clicked.connect(self.optical_effect)
        toggle_optical.stateChanged.connect(toggle_optical.start_transition)
        toggle_optical.setObjectName('toggle_optical')
        # threshold_optical = QDoubleSpinBox()
        # threshold_optical.setValue(1.0)
        # threshold_optical.setSingleStep(0.1)
        grid_optical.addWidget(label_optical, 0, 0, 1, 1)
        # grid_optical.addWidget(threshold_optical, 0, 1, 1, 1)
        grid_optical.addWidget(toggle_optical, 0, 2, 1, 1)
        self.grid_supervised.addLayout(grid_optical, 3, 0, 1, 2)

    ##################################################################################################

    def type_supervised(self):
        # radio group for the type of each curve
        group_button_type = QGroupBox('Pull: Type?')
        group_type = QButtonGroup()
        vbox_type = QHBoxLayout()

        type_nad = QRadioButton("NAD")
        type_nad.setToolTip('No Adhesion')
        type_ad = QRadioButton("AD")
        type_ad.setToolTip('Adhesion')
        type_ftu = QRadioButton("FTU")
        type_ftu.setToolTip('Finished tube')
        type_itu = QRadioButton("ITU")
        type_itu.setToolTip('Infinite tube')
        type_re = QRadioButton("RE")
        type_re.setToolTip('Rejected')
        group_type.addButton(type_nad)
        group_type.addButton(type_ad)
        group_type.addButton(type_ftu)
        group_type.addButton(type_itu)
        group_type.addButton(type_re)
        vbox_type.addWidget(type_nad)
        vbox_type.addWidget(type_ad)
        vbox_type.addWidget(type_ftu)
        vbox_type.addWidget(type_itu)
        vbox_type.addWidget(type_re)
        group_button_type.setLayout(vbox_type)
        if 'type' in self.current_curve.features:
            type_curve = self.current_curve.features['type']
        else:
            type_curve = self.current_curve.features['automatic_type']
        for button in group_type.buttons():
            if button.text() == type_curve:
                button.setChecked(True)
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(lambda: self.modif_supervised('type'))
        self.grid_supervised.addWidget(group_button_type, 5, 0, 1, 2)

    ##################################################################################################

    def keyPressEvent(self, event):
        super(View, self).keyPressEvent(event)
        self.keyPressed.emit(event)

    ##################################################################################################
    def mousePressEvent(self, event):
        if self.check_graph:
            self.setFocus()

    ##################################################################################################

    def on_key(self, event):
        """
        management of the left and right arrows 
        to navigate between the pages of the show plot

        :parameters:
            event: keypress
                information about the key pressed on the keyboard 
        """
        if self.check_graph:
            self.setFocus()
            if event.key() == Qt.Key_Left:
                self.previous_plot()
            elif event.key() == Qt.Key_Right:
                self.next_plot()
            elif event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
                self.change()

    ##################################################################################################

    def next_plot(self):
        """
        allows to advance one page in the plots
        """
        if self.n+1 < self.length_list_curve:
            self.n += 1
            self.current_curve.output['treat_supervised'] = True
            self.check_toggle = False
            self.show_graphic()

    ##################################################################################################

    def previous_plot(self):
        """
        allows you to go back one page in the plots
        """
        if self.n != 0:
            self.n -= 1
            self.current_curve.output['treat_supervised'] = False
            self.check_toggle = False
            self.show_graphic()

    #############################################################################################

    def modif_supervised(self, name_group):
        """
        Action when clicking on the checkboxes
        """
        sender = self.sender()
        if self.dict_supervised[self.current_curve.file] == sender.parent().parent():
            self.controller.add_feature(
                self.current_curve.file, name_group, sender.text())
        self.show_graphic()

    ############################################################################################

    def option_supervised(self, option):
        """
        Action when clicking on the checkboxes in the supervised menu
        """
        self.check_supervised = option
        self.count_select_plot = 0
        self.clear()
        self.show_graphic()

    ###########################################################################################

    def optical_effect(self):
        """
        Creation of the manual optical correction interface
        """
        try:
            self.toggle = self.sender()
            self.check_toggle = self.toggle.isChecked()
            self.intreval_optical_effect = []
            if self.check_toggle:
                if self.dict_supervised[self.current_curve.file] == self.sender().parent():
                    # if len(self.current_curve.dict_segments) == 2:
                    fig = Figure()
                    self.dict_fig_open[self.current_curve.file] = fig
                    fig = self.current_curve.correction_optical_effect_object.manual_correction(
                        fig, self.methods['factor_noise'])
                    fig.canvas.mpl_connect('pick_event', self.data_select)
                    fig.canvas.mpl_connect(
                        'close_event', self.close_window_optical)
                    self.graph_view = GraphView()
                    canvas = FigureCanvasQTAgg(fig)
                    toolbar_optical = NavigationToolbar2QT(
                        canvas, self.graph_view)
                    button_accept_correction = QPushButton('Accept correction')
                    self.graph_view.main_layout.addWidget(
                        toolbar_optical, 0, 0, 1, 1)
                    self.graph_view.main_layout.addWidget(
                        canvas, 1, 0, 4, 2)
                    self.graph_view.main_layout.addWidget(
                        button_accept_correction, 0, 1, 1, 1)
                    button_accept_correction.clicked.connect(
                        self.accept_manual_correction)
                    button_accept_correction.hide()
                    self.graph_view.showMaximized()
            else:
                plt.close()
            self.setFocus()
        except:
            print('Index error')

    ###########################################################################################

    def data_select(self, event):
        """
        pick event management to select the range of data to correct

        :parameters:
            event: click event to select the indexes of the data range
        """
        if isinstance(event.artist, Line2D):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            self.intreval_optical_effect.append(ind[0])
            if len(self.intreval_optical_effect) > 2:
                self.dict_fig_open[self.current_curve.file].clear()
                self.dict_fig_open[self.current_curve.file] = self.current_curve.correction_optical_effect_object.manual_correction(
                    self.dict_fig_open[self.current_curve.file], self.methods['factor_noise'])
                for child in self.graph_view.children():
                    if isinstance(child, QPushButton):
                        child.hide()
                self.toggle.setChecked(True)
            if len(self.intreval_optical_effect) <= 2:
                ax = self.dict_fig_open[self.current_curve.file].axes[0]
                ax.plot(xdata[ind[0]], ydata[ind[0]],
                        marker='D', color='orange')
                plt.draw()
                if len(self.intreval_optical_effect) == 2:
                    self.current_curve.correction_optical_effect_object.correction_optical_effect(
                        self.intreval_optical_effect, self.dict_fig_open[self.current_curve.file])
                    for child in self.graph_view.children():
                        if isinstance(child, QPushButton):
                            child.show()
            if len(self.intreval_optical_effect) > 2:
                self.intreval_optical_effect = []
            canvas = FigureCanvasQTAgg(
                self.dict_fig_open[self.current_curve.file])
            self.graph_view.main_layout.addWidget(canvas, 1, 0, 4, 2)
            self.graph_view.showMaximized()

    #####################################################################################################

    def accept_manual_correction(self):
        """
        management of the actions during the accptation of the manual optical correction
        """
        self.current_curve.correction_optical_effect_object.accept_correction()
        self.graph_view.close()
        self.current_curve.analyzed_curve(self.methods, True)
        self.current_curve.features['optical_state'] = "Manual_correction"
        self.show_graphic()

    #####################################################################################################

    def show_bilan(self):
        """
        Creation of the diagram window to illustrate the results of the analysis
        """
        self.check_bilan = True
        today = str(date.today())
        self.graph_bilan = GraphView()
        hbox_bilan = QHBoxLayout()
        frame = QFrame()
        label_day = QLabel('Date: ' + today +  '\n' + 'Condition: ' + str(self.methods['condition']) + '\n' + 'Drug: ' + str(self.methods['drug']))
        # label_condition = QLabel(
        #     'Condition: ' + str(self.methods['condition']))
        # label_drug = QLabel()
        nb_beads, nb_cells, nb_couples = self.controller.count_cell_bead()
        label_nb_bead = QLabel('Nb beads: ' + str(nb_beads) + '\nNb cells: ' +
                               str(nb_cells) + '\nNb couples: ' + str(nb_couples))
        label_type_files = QLabel('Nb txt files: ' + str(
            self.controller.dict_type_files['txt']) + '\nNb jpk files: ' + str(self.controller.dict_type_files['jpk']))
        hbox_bilan.addWidget(label_day)
        # hbox_bilan.addWidget(label_condition)
        # hbox_bilan.addWidget(label_drug)
        hbox_bilan.addWidget(label_nb_bead)
        hbox_bilan.addWidget(label_type_files)
        frame.setLayout(hbox_bilan)
        frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        frame.setLineWidth(3)
        frame.setMidLineWidth(3)
        fig = Figure()
        fig = self.controller.piechart(fig)
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        self.graph_bilan.main_layout.addWidget(toolbar, 0, 0, 1, 1)
        self.graph_bilan.main_layout.addWidget(frame, 0, 1, 1, 3)
        
        self.graph_bilan.main_layout.addWidget(canvas, 1, 0, 6, 4)

        self.graph_bilan.showMaximized()

    ###########################################################################################

    def close_window_optical(self, event):
        """
        Management of the closing of the window of manual optical correction
        """
        # self.setFocus()
        if self.check_toggle:
            self.toggle.setChecked(False)
            self.check_toggle = False
            del self.dict_fig_open[self.current_curve.file]

    ################################################################################################
    def save(self):
        """
        Generates the output file when Save is clicked
        """
        directory = QFileDialog.getExistingDirectory(
            self, "Open folder", ".", QFileDialog.ShowDirsOnly)
        today = str(date.today())
        time_today = str(datetime.now().time().replace(
            microsecond=0)).replace(':', '-')
        if self.check_graph:
            self.current_curve.output['treat_supervised'] = True
        self.controller.output_save(directory)
        methods = {}
        methods['methods'] = self.methods
        output_methods = pd.DataFrame()
        output_methods = output_methods.from_dict(methods, orient='index')
        list_labels_methods = ['condition', 'drug', 'bead_radius', 'model', 'eta', 'pulling_length', 'threshold_align',
                               'jump_force', 'jump_distance', 'jump_point', 'factor_noise', 'optical']
        output_methods = output_methods[list_labels_methods]
        output_methods.to_csv(directory + sep + 'methods_' + today + '_' +
                              time_today + '.tsv', sep='\t', encoding='utf-8', na_rep="NaN")

        if self.check_graph or self.save_table.isChecked():
            self.close()

        return directory

    ###########################################################################################
    def save_graph(self):
        """
        Management of the saving of the graphics step by step on the supervised part
        """
        self.nb_save_graph += 1
        name_img = ""
        if self.nb_save_graph == 1:
            self.directory_graphs = QFileDialog.getExistingDirectory(
                self, "Open folder", ".", QFileDialog.ShowDirsOnly)
        if self.abscissa_curve:
            name_img = self.controller.save_plot_step(
                self.fig, self.current_curve, 'global', self.directory_graphs)
        else:
            name_img = self.controller.save_plot_step(
                self.fig, self.current_curve, 'distance', self.directory_graphs)
        #label_save_graphs = QLabel('graph registered under the name: \n' + name_img)
        label_save_graphs = QLabel('well registered graphic')
        self.main_layout.addWidget(label_save_graphs, 3, 6, 1, 1)
        self.setFocus()

    ##########################################################################################

    def save_and_save_graphs(self):
        """
        Allows on the first interface to save all the graphics and the output file
        """
        length = str(len(self.controller.create_list_for_graphs()))
        directory = self.save()
        self.msgBox.show()
        self.msgBox.setWindowTitle("save_graph")
        self.msgBox.setText("Loading...\n" + "0/" + length)
        loop = QEventLoop()
        QTimer.singleShot(5, loop.quit)
        loop.exec_()
        sleep(2.0)
        self.controller.save_graphs(directory)
        self.close()

    #########################################################################################

    def clear(self):
        """
        Allows you to delete all the widgets present on the main grid of the interface
        """
        while self.main_layout.count():
            child = self.main_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    ########################################################################################

    def closeEvent(self, event):
        """
        Management of the closing of the main window
        """
        if self.info:
            self.info.close()
        if self.check_bilan:
            self.graph_bilan.close()
        plt.close('all')


if __name__ == "__main__":
    view = View()
    view.show()
