"""
Class Controller
"""
import sys
from os import sep
from pathlib import Path
from shutil import copy
from time import time
from datetime import date, datetime
import pandas as pd
import numpy as np
from pandas.core.tools.numeric import to_numeric
from model.class_curve import Curve
from model.class_segment_curve import Segment
from extractor.extractor_JPK import JPKFile
from matplotlib.figure import Figure
from matplotlib import gridspec
import re
import argparse
from math import ceil, floor


class Controller:
    """
    Instantiates "controller" objects for processing optical curves
    """

    def __init__(self, view=None, path_files=None):
        """
        initialization of the basic attributes of the control

        :parameters:
            view: Object
                interface object
            path_files: str
                path of the folder containing the curves to be analyzed
        """
        self.view = view
        self.files = []
        self.dict_curve = {}
        self.test = None
        self.output = pd.DataFrame()
        if path_files is not None:
            self.set_list_files(path_files)
            methods = {'threshold_align': 30, 'pulling_length': 50, 'model': 'linear', 'eta': 0.5, 'bead_radius': 1, 'factor_noise': 5,
                       'jump_force': 5, 'jump_point': 200, 'jump_distance': 200, 'drug': 'NaN', 'condition': 'NaN', 'optical': None}
            self.set_list_curve(methods)

    ##############################################################################################

    def set_view(self, view):
        """
        setter of the view attribute

        :parameters:
            view: Object
                interface object
        """
        self.view = view

    ##############################################################################################

    # def set_list_curve(self, threshold_align=30, pulling_length=50, model='linear', eta=0.5, bead_ray=1, tolerance=5, jump_force=5, jump_points=200, jump_distance=200, drug='', condition='aCD3', optical_correction=None):
    def set_list_curve(self, methods):
        """
        creation of the curve list according to the file extension and its conformity
        """
        self.dict_type_files = {'txt': 0, 'jpk': 0}
        self.list_file_imcomplete = set()
        for index_file in range(0, len(self.files), 1):
            # print(index_file)
            # print(self.files[index_file])
            new_curve = None
            type_file = self.files[index_file].split('.')[-1]
            name_file = self.files[index_file].split(
                sep)[-1]  # .split('.')[0:-1]
            #name_file = '.'.join(name_file)
            regex = re.match("^b[1-9]c[1-9][a-z]{0,2}-", name_file)
            name_file = name_file.split('-')
            name_file = str(name_file[0][0:4]) + '-' + '-'.join(name_file[1:])
            nb = str(index_file+1) + "/" + str(len(self.files))
            if name_file not in self.dict_curve:
                check_incomplete = False
                try:
                    if type_file == 'txt' and regex:
                        new_curve, check_incomplete = Controller.open_file(
                            self.files[index_file], methods['threshold_align'], methods['pulling_length'])
                        self.dict_type_files['txt'] += 1
                    elif type_file == 'jpk-nt-force':
                        new_curve, check_incomplete = Controller.create_object_curve(
                            self.files[index_file], methods['threshold_align'], methods['pulling_length'])
                        self.dict_type_files['jpk'] += 1
                    else:
                        print(
                            '\n===============================================================================')
                        print(self.files[index_file].split(sep)[-1])
                        print(
                            '===============================================================================')
                        print('non-conforming extension.')
                except:
                    print("The curve is not conform")

                if check_incomplete:
                    self.list_file_imcomplete.add(
                        self.files[index_file].split(sep)[-1])
                if new_curve is not None:
                    if new_curve.check_incomplete:
                        if type_file == 'jpk-nt-force':
                            type_file = type_file.split('-')[0]
                        Controller.file_incomplete_rejected(
                            type_file, self.files[index_file])
                        self.list_file_imcomplete.add(
                            self.files[index_file].split(sep)[-1])
                    else:
                        self.dict_curve[new_curve.file] = new_curve
                        new_curve.analyzed_curve(
                            methods, False)
                        new_curve.features['type'] = new_curve.features['automatic_type']
            else:
                print(
                    '\n===============================================================================')
                print(self.files[index_file].split(sep)[-1])
                print(
                    '===============================================================================')
                print('files already processed')
            if self.view is not None:
                self.view.info_processing(nb, len(self.files))

    ##################################################################################################################

    def create_list_for_graphs(self):
        """
        creation of the list separating the curves in list of 2 for the display of the graphs sequentially

        :return:
            list_curves_for_graphs: list
                list of list with the curve to analysis for display graph
        """
        list_curves_for_graphs = []
        nb_graph = 2
        for i in range(0, len(self.dict_curve), nb_graph):
            list_curves_for_graphs.append(
                list(self.dict_curve.values())[i:i+nb_graph])
        return list_curves_for_graphs

    ##############################################################################################
    def show_plot(self, n, abscissa_curve='distance'):
        """
        creation of graphs by curves 2 per page maximum

        :parameters:
            n: int
                index of the curve list
            abscissa_curve: str
                name of the data for the abscissa of the curve

        :return:
            fig: Object
                figure to be displayed on the interface canvas
            curve: Object
                the current curve to display graphs

        """
        nb_graph = 1
        list_curves_for_graphs = list(self.dict_curve.values())
        if len(list_curves_for_graphs) > 0:
            curve = list_curves_for_graphs[n]
            fig = Figure()
            graph_position = 1
            max_handle = [[], []]
            if abscissa_curve == 'time':
                graph_position = self.plot_time(curve, fig, graph_position)
            else:
                for segment in curve.dict_segments.values():
                    if segment.header_segment['segment-settings.style'] == "motion":
                        if segment.name == 'Press':
                            time_wait = 'Waiting Time: ' + \
                                curve.parameters_header['header_global']['settings.segment.1.duration'] + ' s'
                            title = segment.name + ' segment, ' + time_wait
                        elif segment.name == 'Pull':
                            #calcul_threshold = curve.features['tolerance'] * curve.graphics['threshold_pull'][0]/curve.features['tolerance']
                            title = f"{segment.name} segment"
                        position = int(str(nb_graph) + '2' +
                                       str(graph_position))
                        ax = fig.add_subplot(position, title=title)
                        graph_position = self.plot_distance(
                            curve, segment, ax, graph_position)
                        if segment.name == 'Press':
                            ax.legend(loc="lower left")
                        elif segment.name == 'Pull':
                            ax.legend(loc="lower right")
        fig.subplots_adjust(wspace=0.3, hspace=0.5)
        fig.tight_layout()
        return fig, curve

    ########################################################################################################################################
    def plot_distance(self, curve, segment, ax, graph_position):
        """
        allows the display of graphs as a function of the distance on the 2 segments

        :parameters:
            curve: Object
                curve under analysis
            segment: Object
                segment motion of the curve
            ax: Object
                axis of the figure where to put the graph
            graph_position: int
                position of the graph on the figure

        :return:
            graph_position: int
                back to the following chart positi
        """
        main_axis = curve.features['main_axis']['axe']
        force_data = segment.corrected_data[main_axis + 'Signal1']
        distance_data = segment.corrected_data['distance']
        time_data = segment.corrected_data['seriesTime']
        fitted_data = curve.graphics['fitted_' + segment.name]
        ax.plot(distance_data, force_data, color="#c2a5cf")
        ax.plot(distance_data, fitted_data, color="#5aae61",
                label=curve.features['model'] + " fit")
        #y_smooth = curve.graphics['y_smooth_' + segment.name]
        y_smooth = curve.smooth(force_data, time_data, 151, 2)
        ax.plot(distance_data, y_smooth, color="#80cdc1")
        index_x_0 = 0
        if segment.name == 'Press':
            threshold_press = curve.graphics['threshold_press']
            threshold_press_neg = np.negative(threshold_press)
            calcul_threshold = curve.features['tolerance'] * \
                curve.graphics['threshold_press'][0] / \
                curve.features['tolerance']
            legend_threshold = f"{curve.features['tolerance']} x STD = +/- {calcul_threshold:.2f} pN"
            ax.plot(distance_data, threshold_press, color='blue',
                    label=legend_threshold, ls='-.', alpha=0.5)
            ax.plot(distance_data, threshold_press_neg,
                    color='blue', alpha=0.5, ls='-.')
            if self.view.methods['optical'] == "Correction":
                index_x_0 = curve.features['contact_theorical_press']['index']
            else:
                index_x_0 = curve.features['contact_point']['index']
            index_max = curve.features["force_min_press"]['index']
            max_curve = curve.features["force_min_curve"]['value']
            ax.plot(distance_data[index_max], force_data[index_max],
                    color='#1b7837', marker='o', label='max')
            ax.plot(distance_data[index_max], max_curve,
                    color='yellow', marker='o', label='max_curve')
            ax.set_ylabel("Force (pN)")
        elif segment.name == 'Pull':
            calcul_threshold = curve.features['tolerance'] * \
                curve.graphics['threshold_pull'][0]/curve.features['tolerance']
            legend_threshold = f"{curve.features['tolerance']} x STD = +/- {calcul_threshold:.2f} pN"
            threshold_pull = curve.graphics['threshold_pull']
            threshold_pull_neg = np.negative(threshold_pull)
            ax.plot(distance_data, threshold_pull, color='blue',
                    label=legend_threshold, ls='-.', alpha=0.5)
            ax.plot(distance_data, threshold_pull_neg,
                    color='blue', alpha=0.5, ls='-.')
            if self.view.methods['optical'] == "Correction":
                index_x_0 = curve.features['contact_theorical_pull']['index']
            else:
                index_x_0 = curve.features['point_release']['index']
            index_max = curve.features['force_max_pull']['index']
            if curve.features['automatic_type'] != 'NAD' and curve.features['automatic_type'] != 'RE':
                if 'milieu_pente' in curve.graphics:
                    index_milieu_derive = curve.graphics['milieu_pente']
                    index_max_derive = curve.graphics['index_max_derive']
                    index_min_derive = curve.graphics['min_derive']
                    ax.plot(distance_data[index_milieu_derive],
                            force_data[index_milieu_derive], marker='D', color='cyan')
                    ax.plot(distance_data[index_max_derive],
                            force_data[index_max_derive], marker='D', color='red')
                    ax.plot(distance_data[index_min_derive],
                            force_data[index_min_derive], marker='D', color='black')
            if curve.features['point_return_endline']['index'] != "NaN":
                index_return = curve.features['point_return_endline']['index']
                index_transition = curve.features['point_transition']['index']
                ax.plot(distance_data[index_return], force_data[index_return],
                        color='#50441b', marker='o', label='return')
                ax.plot(distance_data[index_transition], force_data[index_transition],
                        color='#1E90FF', marker='o', label='transition')
            if 'type' in curve.features:
                if curve.features['type'] != 'NAD':
                    ax.plot(distance_data[index_max], force_data[index_max],
                            color='#1b7837', marker='o', label='max')
            else:
                if curve.features['automatic_type'] != 'NAD':
                    ax.plot(distance_data[index_max], force_data[index_max],
                            color='#1b7837', marker='o', label='max')
        ax.plot(distance_data[index_x_0], force_data[index_x_0],
                color='#762a83', marker='o', label='contact/release')
        ax.set_ylim(curve.features['force_min_curve']['value']-2,
                    curve.features['force_max_curve']['value'] + 2)
        ax.set_xlabel("Corrected distance (nm)")
        graph_position += 1

        return graph_position
    ###########################################################################################################################################

    def plot_time(self, curve, fig, graph_position):
        """
        allows the display of graphs as a function of time on the 3 axes

        :parameters:
            curve: Object
                curve under analysis
            fig: Object
                figure where to insert the graphics
            graph_position: int
                position of the graph on the figure

        :return:
            graph_position: int
                back to the following chart position
        """
        main_axis = curve.features['main_axis']['axe']
        data_total = curve.retrieve_data_curve('data_corrected')
        segment_pull = curve.dict_segments['Pull']
        force_data_pull = segment_pull.corrected_data[main_axis + 'Signal1']
        time_data_pull = segment_pull.corrected_data['seriesTime']
        threshold_align = curve.graphics['threshold_press'][0]
        threshold_line_pos = np.full(len(data_total), threshold_align)
        threshold_line_neg = np.negative(
            np.full(len(data_total['seriesTime']), threshold_align))
        ax1 = fig.add_subplot(111, title="Main axis: " + main_axis)
        data_total.plot(kind="line", x='seriesTime', y=main_axis + 'Signal1',
                        xlabel='time (s)', ylabel='Force (pN)', ax=ax1, color='green', alpha=0.5, legend=None)
        ax1.plot(data_total['seriesTime'], np.zeros(len(data_total[main_axis + 'Signal1'])),
                 color='green', alpha=0.75)
        ax1.plot(data_total['seriesTime'], threshold_line_pos,
                 color='blue', label='threshold', ls='-.', alpha=0.5)
        ax1.plot(data_total['seriesTime'], threshold_line_neg,
                 color='blue', ls='-.', alpha=0.5)
        index_contact = curve.features['contact_point']['index']
        index_min_press = curve.features['force_min_press']['index']
        index_release = curve.features['point_release']['index']
        index_max = curve.features['force_max_pull']['index']
        if curve.features['point_transition']['index'] != 'NaN' and curve.features['point_return_endline']['index'] != 'NaN':
            index_trasition = curve.features['point_transition']['index']
            index_return_endline = curve.features['point_return_endline']['index']
            ax1.plot(time_data_pull[index_trasition], force_data_pull[index_trasition],
                     marker='o', color='#1E90FF', label='transition point')
            ax1.plot(time_data_pull[index_return_endline], force_data_pull[index_return_endline],
                     marker='o', color='#50441b', label='return endline')
        ax1.plot(data_total['time'][index_contact], data_total[main_axis + 'Signal1']
                 [index_contact], marker='o', color='#762a83', label='contact point')
        ax1.plot(data_total['time'][index_min_press], data_total[main_axis + 'Signal1']
                 [index_min_press], marker='o', color='#8b5847', label='min press')
        ax1.plot(curve.features['time_min_curve']['value'], curve.features['force_min_curve']
                 ['value'], marker='o', label='min curve', color='red')
        ax1.plot(time_data_pull[index_release], force_data_pull[index_release],
                 marker='o', color='#762a83', label='release point')
        ax1.plot(time_data_pull[index_max], force_data_pull[index_max],
                 marker='o', color='#1b7837', label='max curve')
        ax1.legend(loc="lower left", ncol=2)
        return graph_position

    ###########################################################################################################################################
    def global_plot(self, n):
        #nb_graph = 1
        list_curves_for_graphs = list(self.dict_curve.values())
        fig = Figure()
        if len(list_curves_for_graphs) > 0:
            curve = list_curves_for_graphs[n]
            main_axis = curve.features['main_axis']['axe']
            data_total = curve.retrieve_data_curve('data_corrected')
            # print(data_total[main_axis + 'Signal1'])
            threshold_align = curve.graphics['threshold alignement']
            threshold_line_pos = np.full(len(data_total), threshold_align)
            threshold_line_neg = np.negative(
                np.full(len(data_total['seriesTime']), threshold_align))

            gs = gridspec.GridSpec(7, 10)
            ax1 = fig.add_subplot(
                gs[0:2, 0:4], title="Main axis: " + main_axis)
            # data_total.plot(kind="line", x='seriesTime', y=main_axis + 'Signal1', \
            #     xlabel='time (s)', ylabel='Force (pN)', ax=ax1, color='green', alpha=0.5, legend=None)
            ax1.plot(data_total['seriesTime'], data_total[main_axis +
                     'Signal1'], color='green', alpha=0.5)
            ax1.plot(data_total['seriesTime'], np.zeros(len(data_total[main_axis + 'Signal1'])),
                     color='black', alpha=0.75)
            # ax1.plot(curve.features['time_min_curve']['value'],
            #          curve.features['force_min_curve']['value'], marker='o', label='force_min')
            scale = 0
            if abs(ax1.get_ylim()[0]) < ax1.get_ylim()[1]:
                scale = abs(ax1.get_ylim()[1])
            else:
                scale = abs(ax1.get_ylim()[0])
            ax1.set_xlabel('time (s)')
            ax1.set_ylabel('Force (pN)')
            ax1.set_ylim(-scale - 3, scale + 3)
            # ax1.legend(loc='upper left')
            if main_axis == 'x':
                ax2 = fig.add_subplot(gs[0:2, 5:7], title="Axis: y")
                ax2.plot(data_total['seriesTime'], data_total['ySignal1'],
                         color='grey', alpha=0.5)
                ax2.set_xlabel('time (s)')
                ax2.plot(data_total['seriesTime'], np.zeros(len(data_total['ySignal1'])),
                         color='black', alpha=0.75)
                ax2.plot(data_total['seriesTime'], threshold_line_pos,
                         color='blue', ls='-.', alpha=0.5)
                ax2.plot(data_total['seriesTime'], threshold_line_neg,
                         color='blue', ls='-.', alpha=0.5)
                ax2.set_ylim(ax1.get_ylim())
            elif main_axis == 'y':
                ax2 = fig.add_subplot(gs[0:2, 5:7], title="Axis: x")
                ax2.plot(data_total['seriesTime'], data_total['xSignal1'],
                         color='grey', alpha=0.5)
                ax2.set_xlabel('time (s)')
                ax2.plot(data_total['seriesTime'], np.zeros(len(data_total['xSignal1'])),
                         color='black', alpha=0.75)
                ax2.plot(data_total['seriesTime'], threshold_line_pos,
                         color='blue', ls='-.', alpha=0.5)
                ax2.plot(data_total['seriesTime'], threshold_line_neg,
                         color='blue', ls='-.', alpha=0.5)
                ax2.set_ylim(ax1.get_ylim())
            ax3 = fig.add_subplot(gs[0:2, 8:10], title="Axis: z")
            ax3.plot(data_total['seriesTime'], data_total['zSignal1'],
                     color='grey', alpha=0.5)
            ax3.set_xlabel('time (s)')
            ax3.plot(data_total['seriesTime'], np.zeros(len(data_total['zSignal1'])),
                     color='black', alpha=0.75)
            ax3.plot(data_total['seriesTime'], threshold_line_pos,
                     color='blue', ls='-.', alpha=0.5)
            ax3.plot(data_total['seriesTime'], threshold_line_neg,
                     color='blue', ls='-.', alpha=0.5)
            ax3.set_ylim(ax1.get_ylim())
        # version avec tous les segments
            length = len(curve.dict_segments.values())
            position_start_graph = 0
            num_segment = 0
            for segment in curve.dict_segments.values():
                if not segment.name.startswith('Wait'):
                    position_end_graph = 0
                    if num_segment == 0:
                        position_end_graph = ceil(
                            position_start_graph + 10/length - 1)
                    else:
                        position_end_graph = 10
                    ax4 = fig.add_subplot(
                        gs[4:7, position_start_graph:position_end_graph])
                    position_start_graph = position_end_graph + 1
                    ax4.plot(
                        segment.corrected_data['distance'], segment.corrected_data[main_axis + 'Signal1'], color="#c2a5cf")
                    ax4.plot(segment.corrected_data['distance'], np.zeros(
                        len(segment.corrected_data[main_axis + 'Signal1'])), color='black', alpha=0.75)
                    ax4.set_xlabel('Corrected distance (nm)')
                else:
                    position_end_graph = floor(
                        position_start_graph + 10/length - 1)
                    ax4 = fig.add_subplot(
                        gs[4:7, position_start_graph:position_end_graph])
                    ax4.plot(
                        segment.corrected_data['seriesTime'], segment.corrected_data[main_axis + 'Signal1'], color="#34b6cf")
                    position_start_graph = position_end_graph + 1
                    ax4.set_xlabel('time (s)')
                if segment.name == 'Press':
                    ax4.set_ylabel('Force (pN)')
                ax4.set_title(segment.name + ' Segment')
                ax4.set_ylim(curve.features['force_min_curve']['value'] -
                             1, curve.features['force_max_pull']['value'] + 2)
                num_segment += 1
        return fig, curve

    ###########################################################################################################################################

    def save_plot_step(self, fig, curve, abscissa_curve, directory_graphs):
        """
        recording of step-by-step graphics. Only the one displayed is saved as a png image

        :parameters:
            fig: object
                figure with graphics
            curve: Object
                curve object corresponding to the graphs 
            abscissa_curve: str
                name of the data for the abscissa of the curve
        :return:
            name_img: str
                name of the created image

        """
        name_curve = ""
        today = date.today()
        count = 1
        name_curve = curve.file
        count += 1
        path_graphs = Path(directory_graphs + sep + 'graphs_' + str(today))
        path_graphs.mkdir(parents=True, exist_ok=True)
        name_img = ""
        if abscissa_curve == 'distance':
            name_img = 'fig_' + name_curve + '_' + str(today) + '_distance.png'
        else:
            name_img = 'fig_' + name_curve + '_' + str(today) + '_time.png'
        fig.savefig(path_graphs.__str__() + sep +
                    name_img, bbox_inches='tight')

        return name_img

    ##############################################################################################

    def add_feature(self, name_curve, add_key, add_value):
        """
        Allows to add data to the parameter dictionary of each curve

        :parameters:
            name_curve: str
                file name of the curve
            add_key: str
                key to add in the features
            add_value: struct
                any data structure storing important information
        """
        for curve in self.dict_curve.values():
            if curve.file == name_curve:
                curve.add_feature(add_key, add_value)
            if add_key not in curve.features:
                if add_key == 'type':
                    curve.add_feature(
                        add_key, curve.features['automatic_type'])
                elif add_key == 'valid_fit_press' or add_key == 'valid_fit_pull':
                    curve.add_feature(add_key, 'False')
                elif add_key == 'AL':
                    curve.add_feature(add_key, curve.features['AL']['AL'])

    ##############################################################################################

    def clear(self):
        """
        Reset of the controller data structure
        """
        self.files = []
        self.dict_curve = {}
        self.test = None
        self.output = pd.DataFrame()

    ################################################################################################

    def set_list_files(self, path):
        """
        Creation of a file list based on a given directory

        :parameters:
            path: str
                path to a study folder
        """
        path_repository_study = Path(path)
        # Allows you to retrieve all the file names of the directory and store them in a list
        if path_repository_study.is_dir():
            for element in path_repository_study.iterdir():
                if element.is_file():
                    self.files.append(element.__str__())
                elif element.is_dir():
                    self.set_list_files(element)
        else:
            self.files = [path_repository_study.__str__()]

    ##############################################################################################

    @staticmethod
    def parsing_generic(file):
        """
        Functioning to parse the headers of jpk_nt_force text files.

        :parameter:
            file: str
                file to parse
        :return:
            header: dict
                dictionary retrieving all the information from the parsed header
        """
        line = file.readline()
        while not line.startswith("# "):
            line = file.readline()
        header = {}
        while line != '#\n':
            if ": " in line:
                line = line.split(": ")
                if line[1].strip() != '':
                    if line[0].replace("# ", "") not in header:
                        header[line[0].replace("# ", "")] = line[1].strip()
            line = file.readline()
        return header

    ##############################################################################################
    @staticmethod
    def parsing_data(file, endfile=False):
        """
        Function that parses the data to produce a matrix

        :parameter:
            file: str
                file to parse
            endfile: bool
                boolean to know if we are at the end of the file after this data set
        :return:
            data: matrix
                list of data lists per row of 15 columns
        """
        data = []
        if endfile:
            for line in file:
                line = line.strip().split(" ")
                data.append(line)
        else:
            line = file.readline()
            while line not in ('#\n', '\n'):
                if line != "":
                    line = line.strip().split(" ")
                    data.append(line)
                    line = file.readline()
        return data

    ##############################################################################################
    @staticmethod
    def management_data(file, nb_segments, num_segment, header_global):
        """
        allows to manage the different cases of segmentation of the curves

        :parameter:
            file: str
                file to process
            nb_segments: int
                number segments in the file
            num_segment: int
                segment number in progress
            header_global: dict
                global information for the curve file 
        :return:
            segment: object
                a segment object with 1 header dictionary and 1 dataframe
        """
        name_segment = ""
        list_name_motion = ["Press", "Wait", "Pull"]
        info_segment = Controller.parsing_generic(file)
        table_parameters = Controller.parsing_generic(file)
        if num_segment < nb_segments-1:
            data_segment = Controller.parsing_data(file)
        else:
            data_segment = Controller.parsing_data(file, True)
        settings_segment = 'settings.segment.' + str(num_segment)
        if header_global[settings_segment + '.style'] == "motion":
            name_segment = list_name_motion[num_segment]
        else:
            name_segment = list_name_motion[1] + str(num_segment)
        dataframe = pd.DataFrame(
            data_segment, columns=table_parameters["columns"].split(" "))
        dataframe = dataframe.apply(to_numeric)
        segment = Segment(info_segment, dataframe, name_segment)
        return segment

    #############################################################################################
    @staticmethod
    def check_file_incomplete(lines, header_global, nb_segments):
        """
        Checks the file, especially if the segments comply
        with the information present in the file header

        :parameters:
            lines: str
                all lines of the file
            header_global: dict
                dictionary of header information
            nb_segments: int
                number of segments stipulated by the header
        :return:
            check_file: bool
                returns true if the file is not truncated
        """
        check_incomplete = True
        nb_block_file = len(lines.split("\n\n"))
        num_segment = 0
        nb_segment_duration_nulle = 0
        for key, value in header_global.items():
            if key == "settings.segment." + str(num_segment) + ".duration":
                if value == "0.0":
                    nb_segment_duration_nulle += 1
                num_segment += 1
        if nb_segments == nb_block_file:
            check_incomplete = False
        elif nb_segments - nb_segment_duration_nulle == nb_block_file:
            check_incomplete = False
        return check_incomplete

    #############################################################################################
    @staticmethod
    def file_troncated(jpk_object):
        """
        Verification of uninterrupted file during manipulation

        :parameters:
            jpk_object: object
                the object representing the jpk-net-force encryption folder

        :return:
            troncated: bool
                returns true if number of segments not in agreement
                with the initiated one otherwise false
        """
        check_incomplete = False
        expected_nb_segments = int(
            jpk_object.headers['header_global']['settings.segments.size'])
        nb_segments_completed = int(
            jpk_object.headers['header_global']['force-segments.count'])
        nb_segments_pause_null = 0
        if expected_nb_segments != nb_segments_completed:
            for index_segment in range(0, expected_nb_segments-1, 1):
                style = jpk_object.headers['header_global']['settings.segment.'
                                                            + str(index_segment) + '.style']
                if style == 'pause':
                    duration = float(jpk_object.headers['header_global']['settings.segment.'
                                                                         + str(index_segment) + '.duration'])
                    if duration == 0.0:
                        nb_segments_pause_null += 1
            if nb_segments_completed != (expected_nb_segments - nb_segments_pause_null):
                check_incomplete = True
        return check_incomplete

    #############################################################################################

    @staticmethod
    def open_file(file, threshold_align, pulling_length):
        """
        if file .txt
        Processing of the curve file to create an object
        with all the information necessary for its study

        :parameter:
            file: str
                name of the curve file
            threshold_align: int
                percentage of maximum force for misalignment 
        :return:
            new_curve: object
                Curved object with 1 title, 4 dictionaries and 2 dataframes
            check_incomplete: bool
                takes True if the file is incomplete otherwise False
        """
        new_curve = None
        header = {}
        check_incomplete = False
        nb_segments = 0
        dict_segments = {}
        title = file.split(sep)[-1].replace(".txt", "")
        file_curve = file.__str__().split(sep)[-1]  # .replace('.txt', "")
        check_incomplete = False
        with open(file, 'r') as file_study:
            lines = file_study.read()
            file_study.seek(0)
            header['header_global'] = Controller.parsing_generic(file_study)
            nb_segments = int(header['header_global']
                              ["settings.segments.size"])
            header['calibrations'] = Controller.parsing_generic(file_study)
            check_incomplete = Controller.check_file_incomplete(lines, header['header_global'],
                                                                nb_segments)
            if check_incomplete:
                Controller.file_incomplete_rejected("txt", file)
            else:
                num_segment = 0
                while num_segment < nb_segments:
                    segment = Controller.management_data(file_study, nb_segments,
                                                         num_segment, header['header_global'])
                    if num_segment < nb_segments-1:
                        settings_segment = 'settings.segment.' + \
                            str(num_segment + 1)
                        if header['header_global'][settings_segment + '.style'] != "motion":
                            if float(header['header_global'][settings_segment +
                                                             ".duration"]) == 0.0:
                                num_segment += 1
                    num_segment += 1
                    dict_segments[segment.name] = segment
                new_curve = Curve(file_curve, title, header,
                                  dict_segments, pulling_length)
                dict_align = Controller.alignment_curve(
                    file, new_curve, threshold_align)
                new_curve.features['automatic_AL'] = dict_align
                new_curve.features['AL'] = dict_align['AL']

        return new_curve, check_incomplete

    ################################################################################################

    @staticmethod
    def create_object_curve(file, threshold_align, pulling_length):
        """
        Creation of the Curve object after extraction of the data from the jpk-nt-force coded file

        :parameters:
            file: str
                path to the jpk-nt-force folder to extract and transform into a Curve python object
            threshold_align: int
                percentage of maximum force for misalignment

        :return:
            new_curve: Object
                returns our curve object ready for analysis and classification
            check_incomplete: bool
                takes True if the file is incomplete otherwise False
        """
        new_curve = None
        new_jpk = JPKFile(file)
        # .replace('.jpk-nt-force', '')
        file_curve = file.__str__().split(sep)[-1]
        check_incomplete = False
        check_incomplete = Controller.file_troncated(new_jpk)
        if check_incomplete:
            Controller.file_incomplete_rejected("jpk", file)
        else:
            columns = []
            dict_segments = {}
            for key in new_jpk.segments[0].data.keys():
                columns.append(key)
            time_end_segment = 0
            time_step = 0
            list_name_segment = ["Press", "Wait", "Pull"]
            num_segment = 0
            for segment in new_jpk.segments:
                print(len(new_jpk.segments))
                name_segment = ""
                dataframe = pd.DataFrame()
                for index_column in range(0, len(columns), 1):
                    if columns[index_column] == 't':
                        data_time = segment.get_array([columns[index_column]])
                        dataframe['time'] = data_time
                        dataframe['seriesTime'] = dataframe['time'].add(time_end_segment
                                                                        + time_step)
                        if segment.index == 0:
                            time_step = dataframe['time'][1]
                        time_end_segment = dataframe['seriesTime'][len(
                            dataframe['seriesTime'])-1]
                    elif columns[index_column] == 'distance':
                        data_distance = segment.get_array(
                            [columns[index_column]])
                        dataframe['distance'] = data_distance
                    else:
                        data_by_column = segment.get_array(
                            [columns[index_column]])
                        dataframe[columns[index_column]] = data_by_column[:, 0]
                if float(new_jpk.headers["header_global"]["settings.segment."
                                                          + str(num_segment) + ".duration"]) == 0.0:
                    num_segment += 1
                if segment.header['segment-settings.style'] == "motion":
                    if num_segment == 0 or num_segment == 2:
                        print(list_name_segment)
                        print(num_segment)
                        name_segment = list_name_segment[num_segment]
                    else:
                        name_segment = "Next_segment_" + str(num_segment)
                else:
                    name_segment = list_name_segment[num_segment] + \
                        str(num_segment)
                num_segment += 1
                dataframe = dataframe.apply(to_numeric)
                new_segment = Segment(segment.header, dataframe, name_segment)
                dict_segments[new_segment.name] = new_segment
            title = new_jpk.headers['title']
            new_curve = Curve(file_curve, title, new_jpk.headers,
                              dict_segments, pulling_length)
            dict_align = Controller.alignment_curve(
                file, new_curve, threshold_align)
            new_curve.features['automatic_AL'] = dict_align
            new_curve.features['AL'] = dict_align['AL']
        return new_curve, check_incomplete

    ############################################################################################

    @staticmethod
    def file_incomplete_rejected(extension, file):
        """
        Allows you to copy files with incomplete segments to a folder

        ;parameters:
            extension: str
                extension of the curve file (txt or jpk-nt-force)
            file: str
                name of the curve file
        """
        path_dir_incomplete = ""
        if extension == "jpk":
            path_dir_incomplete = Path(
                '..' + sep + 'File_rejected' + sep + 'Incomplete' + sep + 'JPK')
        else:
            path_dir_incomplete = Path(
                '..' + sep + 'File_rejected' + sep + 'Incomplete' + sep + 'TXT')
        path_dir_incomplete.mkdir(parents=True, exist_ok=True)
        file_curve = file.__str__().split(sep)[-1]
        copy(file, str(path_dir_incomplete))
        print("\n")
        print("========================================================================")
        print(file_curve)
        print("========================================================================")
        print("File incomplete")

    ############################################################################################

    @staticmethod
    def alignment_curve(file, new_curve, threshold_align):
        """
        Calls the result of the method of checking the curved object well aligned on the main axis.
        If not then export the file to a rejected directory

        :parameters:
            file: str
                name of the curve file
            new_curve: Object
                curve for the analysis of the alignment
            threshold_align: int
                percentage of maximum force for misalignment
        :return:
            dict_align: dict
                information on the good or bad alignment as well as the secondary axis(es) involved


        """
        dict_align = new_curve.check_alignment_curve(threshold_align)
        if dict_align['AL'] == 'No':
            path_dir_alignment = ""
            name_file = file.split(sep)[-1]
            if name_file.split('.')[-1] == "txt":
                path_dir_alignment = Path(
                    '..' + sep + 'File_rejected' + sep + 'Alignment' + sep + 'TXT')
            elif name_file.split('.')[-1] == "jpk-nt-force":
                path_dir_alignment = Path(
                    '..' + sep + 'File_rejected' + sep + 'Alignment' + sep + 'JPK')
            path_dir_alignment.mkdir(parents=True, exist_ok=True)
            copy(file, str(path_dir_alignment))
        return dict_align

    #############################################################################################

    def save_graphs(self, directory_graphs):
        """
        recording of all the graphs of the analysis

        :parameters:
            directory_graphs: str
                path to which to save the graphs
        """
        #list_curves_for_graphs = self.create_list_for_graphs()
        list_curves_for_graphs = list(self.dict_curve.values())
        for index_list in range(0, len(list_curves_for_graphs), 1):
            fig, curve = self.show_plot(index_list)
            self.save_plot_step(fig, curve, 'distance', directory_graphs)
            fig, curve = self.show_plot(index_list, 'time')
            self.save_plot_step(fig, curve, 'time', directory_graphs)
            nb = str(index_list+1) + "/" + str(len(list_curves_for_graphs))
            self.view.info_processing(nb, len(list_curves_for_graphs))

    ##############################################################################################

    def piechart(self, fig):
        """
        creation of pie charts for the analysis balance window

        :parameters:
            fig: main figure blank to be completed

        :return:
            fig: main figure completed by the different diagrams
        """
        # pie chart incomplete curves
        ax_incomplete = fig.add_subplot(231)
        nb_curves = len(self.dict_curve)
        nb_files = len(self.files)
        dict_type_auto = {'NAD': 0, 'AD': 0, 'FTU': 0, 'ITU': 0, 'RE': 0}
        dict_type_supervised = {'NAD': 0, 'AD': 0, 'FTU': 0, 'ITU': 0, 'RE': 0}
        nb_incomplete = len(self.list_file_imcomplete)
        percent_treat = nb_curves/nb_files * 100
        percent_incomplete = nb_incomplete/nb_files * 100
        percent_no_conforming_files = (
            nb_files-(nb_incomplete + nb_curves))/nb_files * 100

        dict_incomplete = {'INC': f"{percent_incomplete:.2f}",
                           'Treat': f"{percent_treat:.2f}", 'NC': f"{percent_no_conforming_files:.2f}"}
        values = [percent_incomplete, percent_treat,
                  percent_no_conforming_files]
        values_incomplete = [value for value in values if value != 0]
        explode_incomplete = ()
        if len(values_incomplete) == 3:
            explode_incomplete = (0.1, 0, 0.1)
        elif len(values_incomplete) == 2:
            explode_incomplete = (0.1, 0)
        else:
            explode_incomplete = None
        ax_incomplete.pie(values_incomplete, explode=explode_incomplete,
                          autopct=lambda pct: Controller.make_autopct(pct, dict_incomplete), shadow=True)
        title = 'Incomplete \nTotal files: ' + \
            str(nb_files) + '\nTotal_files_curve: ' + \
            str(nb_incomplete + nb_curves)
        ax_incomplete.set_title(title)
        ax_alignment_auto = fig.add_subplot(232)
        ax_alignment_supervised = fig.add_subplot(233)
        ax_classification_before = fig.add_subplot(234)
        ax_classification_after = fig.add_subplot(236)
        nb_alignment_auto = 0
        nb_alignment_supervised = 0
        dict_correction = {'No_correction': 0,
                           'Auto_correction': 0, 'Manual_correction': 0}
        for curve in self.dict_curve.values():
            if curve.features['automatic_AL']['AL'] == 'No':
                nb_alignment_auto += 1
            else:
                if curve.features['automatic_type'] in dict_type_auto:
                    dict_type_auto[curve.features['automatic_type']] += 1
            if curve.features['AL'] == 'No':
                nb_alignment_supervised += 1
            else:
                if 'type' in curve.features:
                    if curve.features['type'] in dict_type_supervised:
                        dict_type_supervised[curve.features['type']] += 1
            dict_correction[curve.features['optical_state']] += 1
        nb_conforming_curves_auto = nb_curves - nb_alignment_auto
        percent_alignment_auto = nb_alignment_auto/nb_curves * 100
        percent_confoming_auto = nb_conforming_curves_auto/nb_curves * 100
        dict_align_auto = {'AL': f"{percent_alignment_auto:.2f}",
                           'CONF': f"{percent_confoming_auto:.2f}"}

        percent_alignment_supervised = nb_alignment_supervised/nb_curves * 100
        nb_conforming_curves_supervised = nb_curves - nb_alignment_supervised
        percent_confoming_supervised = nb_conforming_curves_supervised/nb_curves * 100
        dict_align_supervised = {
            'AL': f"{percent_alignment_supervised:.2f}", 'CONF': f"{percent_confoming_supervised:.2f}"}

        explode_align = (0.1, 0)

        values_align_auto = [percent_alignment_auto, percent_confoming_auto]
        values_align_supervised = [
            percent_alignment_supervised, percent_confoming_supervised]

        ax_alignment_auto.pie(values_align_auto, explode=explode_align, autopct=lambda pct: Controller.make_autopct(
            pct, dict_align_auto), shadow=True, startangle=45)
        ax_alignment_auto.set_title(
            'Automatic Alignment\nTreated curves: ' + str(nb_curves))
        ax_alignment_supervised.pie(values_align_supervised, explode=explode_align, autopct=lambda pct: Controller.make_autopct(
            pct, dict_align_supervised), shadow=True, startangle=45)
        ax_alignment_supervised.set_title(
            'Supervised Alignment\nTreated curves: ' + str(nb_curves))

        percent_NAD_auto = (
            dict_type_auto['NAD']/nb_conforming_curves_auto * 100)
        percent_AD_auto = (
            dict_type_auto['AD']/nb_conforming_curves_auto * 100)
        percent_FTU_auto = (
            dict_type_auto['FTU']/nb_conforming_curves_auto * 100)
        percent_ITU_auto = (
            dict_type_auto['ITU']/nb_conforming_curves_auto * 100)
        percent_RE_auto = (dict_type_auto['RE']/nb_conforming_curves_auto*100)
        dict_classification_auto = {'NAD': f"{percent_NAD_auto:.2f}", 'AD': f"{percent_AD_auto:.2f}",
                                    'FTU': f"{percent_FTU_auto:.2f}", 'ITU': f"{percent_ITU_auto:.2f}", 'RE': f"{percent_RE_auto:.2f}"}
        values = [percent_FTU_auto, percent_NAD_auto,
                  percent_RE_auto, percent_AD_auto, percent_ITU_auto]
        values_auto = [value for value in values if value != 0]
        ax_classification_before.pie(values_auto, autopct=lambda pct: Controller.make_autopct(
            pct, dict_classification_auto), shadow=True, startangle=45)
        ax_classification_before.set_title(
            'Classification before\nConforming curves: ' + str(nb_conforming_curves_auto))

        ax_correction = fig.add_subplot(235)
        percent_no_correction = (
            dict_correction['No_correction']/nb_curves * 100)
        percent_auto_correction = (
            dict_correction['Auto_correction']/nb_curves * 100)
        percent_manual_correction = (
            dict_correction['Manual_correction']/nb_curves * 100)
        dict_correction_percent = {'No_correction': f"{percent_no_correction:.2f}",
                                   'Auto_correction': f"{percent_auto_correction:.2f}", 'Manual_correction': f"{percent_manual_correction:.2f}"}
        values = [percent_no_correction,
                  percent_auto_correction, percent_manual_correction]
        values_correction = [value for value in values if value != 0]
        ax_correction.pie(values_correction, autopct=lambda pct: Controller.make_autopct(
            pct, dict_correction_percent), shadow=True, startangle=45)
        ax_correction.set_title(
            'State Correction\nTreated curves: ' + str(nb_curves))

        percent_NAD_supervised = (
            dict_type_supervised['NAD']/nb_conforming_curves_supervised * 100)
        percent_AD_supervised = (
            dict_type_supervised['AD']/nb_conforming_curves_supervised * 100)
        percent_FTU_supervised = (
            dict_type_supervised['FTU']/nb_conforming_curves_supervised * 100)
        percent_ITU_supervised = (
            dict_type_supervised['ITU']/nb_conforming_curves_supervised * 100)
        percent_RE_supervised = (
            dict_type_supervised['RE']/nb_conforming_curves_supervised*100)
        dict_classification_supervised = {'NAD': f"{percent_NAD_supervised:.2f}", 'AD': f"{percent_AD_supervised:.2f}",
                                          'FTU': f"{percent_FTU_supervised:.2f}", 'ITU': f"{percent_ITU_supervised:.2f}", 'RE': f"{percent_RE_supervised:.2f}"}
        values = [percent_FTU_supervised, percent_NAD_supervised,
                  percent_RE_supervised, percent_AD_supervised, percent_ITU_supervised]
        values_supervised = [value for value in values if value != 0]
        ax_classification_after.pie(values_supervised, autopct=lambda pct: Controller.make_autopct(
            pct, dict_classification_supervised), shadow=True, startangle=45)
        ax_classification_after.set_title(
            'Classification after\nConforming curves: ' + str(nb_conforming_curves_auto))

        return fig
    ##################################################################################################################################

    @staticmethod
    def make_autopct(pct, dico):
        """
        management of the labels of the corners of the diagram

        :parameters:
            dico:dictionary of values and labels of the diagram

        :return:
            pct: percentage
            val: label
        """
        pct = f"{pct:.2f}"
        #pct = np.round(pct, 2)
        list_keys = list(dico.keys())
        list_values = list(dico.values())
        num_key = list_values.index(pct)
        val = list_keys[num_key]
        del dico[val]
        return f"{pct}%\n({val})"
    #########################################################################################################

    def count_cell_bead(self):
        dict_beads = {}
        dict_cells = {}
        dict_couple = {}
        for curve in self.dict_curve.values():
            if curve.output['bead'] in dict_beads:
                dict_beads[curve.output['bead']] += 1
            else:
                dict_beads[curve.output['bead']] = 1
            if curve.output['cell'] in dict_cells:
                dict_cells[curve.output['cell']] += 1
            else:
                dict_cells[curve.output['cell']] = 1
            if curve.output['couple'] in dict_couple:
                dict_couple[curve.output['couple']] += 1
            else:
                dict_couple[curve.output['couple']] = 1
        nb_beads = len(dict_beads)
        nb_cells = len(dict_cells)
        nb_couples = len(dict_couple)
        return nb_beads, nb_cells, nb_couples

    ##############################################################################################

    def output_save(self, path_directory):
        """
        Transformation of the characteristics of each curve into a general dataframe. 
        Writing of this dataframe in a csv file

        :parameters:
            path_directory: str
                name of the folder to save the output
        """
        print("\noutput_save")
        today = str(date.today())
        time_today = str(datetime.now().time().replace(
            microsecond=0)).replace(':', '-')
        if len(self.dict_curve) > 0:
            dict_infos_curves = {}
            for curve in self.dict_curve.values():
                curve.creation_output_curve()
                dict_infos_curves[curve.file] = curve.output
            self.output = self.output.from_dict(
                dict_infos_curves,  orient='index')
            self.output['main_axis'] = self.output['main_axis_sign'] + \
                self.output['main_axis_axe']
            self.output.drop(
                ['main_axis_sign', 'main_axis_axe'], axis=1, inplace=True)
            if 'valid_fit_press' not in self.output:
                self.output['valid_fit_press'] = True
            if 'valid_fit_pull' not in self.output:
                self.output['valid_fit_pull'] = False
            name_parameters = ""
            error_parameters = ''
            if self.output['model'][0] == 'linear':
                name_parameters = 'slope (pN/nm)'
                error_parameters = 'error (pN/nm)'
            elif self.output['model'][0] == 'sphere':
                name_parameters = 'young (Pa)'
                error_parameters = 'error young (Pa)'
            liste_labels = ['treat_supervised', 'automatic_type', 'type', 'automatic_AL', 'AL', 'automatic_AL_axe',
                            'optical_state', 'model', 'Date', 'Hour', 'condition', 'drug', 'tolerance', 'bead', 'cell',
                            'main_axis', 'stiffness (N/m)', 'theorical_contact_force (N)', 'theorical_distance_Press (m)',
                            'theorical_speed_Press (m/s)', 'theorical_freq_Press (Hz)', 'theorical_distance_Pull (m)',
                            'theorical_speed_Pull (m/s)', 'theorical_freq_Pull (Hz)', 'baseline_press', 'std_press',
                            name_parameters, error_parameters, 'contact_point_index', 'contact_point_value',
                            'force_min_press_index', 'force_min_press_value', 'point_release_index',
                            'point_release_value', 'force_max_pull_index', 'force_max_pull_value',
                            'point_return_endline_index', 'point_return_endline_value']
            if len(liste_labels) != len(self.output.columns):
                for label in self.output.columns:
                    if label not in liste_labels:
                        if label == 'type':
                            liste_labels.insert(1, label)
                        elif label.startswith('time_segment_pause'):
                            self.output[label] = self.output[label].replace(
                                np.nan, 0)
                            if label.endswith('Wait1 (s)'):
                                liste_labels.insert(liste_labels.index(
                                    'theorical_freq_Press (Hz)') + 1, label)
                            else:
                                liste_labels.insert(liste_labels.index(
                                    'theorical_freq_Pull (Hz)') + 1, label)
                        else:
                            liste_labels.append(label)
            self.output = self.output[liste_labels]

            self.output.rename(columns={'baseline_press': 'baseline_press (pN)', 'std_press': 'std_press (pN)',
                                        'contact_point_value': 'contact_point_value  (pN)', 'force_min_press_value': 'force_min_press_value (pN)',
                                        'point_release_value': 'point_release_value (pN)', 'force_max_pull_value': 'force_max_pull_value (pN)',
                                        'point_return_endline_value': 'point_return_endline_value (pN)'}, inplace=True)

            for incomplete in self.list_file_imcomplete:
                self.output.loc[incomplete, 'automatic_type'] = 'INC'
                self.output.loc[incomplete, 'type'] = 'INC'
            print(self.output)
            print(self.output['type'])
            print(self.output['automatic_AL'])
            if self.view is None:
                path_directory = Path("Result")
                path_directory.mkdir(parents=True, exist_ok=True)
                path_directory = path_directory.__str__()
            name_file = path_directory + sep + 'output_' + today + '_' + time_today + '.csv'
            self.output.to_csv(name_file, sep='\t',
                               encoding='utf-8', na_rep="NaN")
            return Path(name_file)

    ##############################################################################################


def parse_args():
    """
    function to add command line arguments to run the controller without GUI

    """
    parser = argparse.ArgumentParser(description="Creation curve objects")
    parser.add_argument("-p", "--path", type=str,
                        help="Name of the folder containing the curves", required=True)
    parser.add_argument(
        "-o", "--output", help="Name of the folder where to save the results", required=True)
    parser.add_argument("-m", "--method", type=argparse.FileType('r'),
                        help="path to a method file (.tsv)")
    return parser.parse_args()


if __name__ == "__main__":
    START_TIME = time()
    args = parse_args()
    PATH_FILES = args.path
    OUTPUT_DIRECTORY = args.output
    controller = Controller(None, PATH_FILES)
    controller.output_save(OUTPUT_DIRECTORY)
    print("--- %s seconds ---" % (time() - START_TIME))
