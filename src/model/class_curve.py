"""
File describing the instance class of the curved objects
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from lmfit import Model
from model.class_optical_effect import OpticalEffect


class Curve:
    """
    Class that instantiates curved objects
    """
    # pylint: disable=unbalanced-tuple-unpacking

    def __init__(self, file, title, header, dict_segments, pulling_length):
        """
        Initialization attributes of the object and launch the functions
        """
        self.file = file
        bead = title.split("-")[0][0:2]
        cell = title.split("-")[0][2:4]
        couple = title.split("-")[0][:4]
        self.parameters_header = header
        self.dict_segments = dict_segments  # Liste d'objets segments
        self.features = {}
        self.graphics = {}
        self.output = {'bead': bead, 'cell': cell, 'couple': couple}
        self.output['treat_supervised'] = False
        self.message = ""

        self.message += "\n========================================================================\n"

        self.message += self.file
        self.message += "\n========================================================================\n"
        # self.delta_t()
        self.identification_main_axis()
        self.normalization_data()
        self.correction_optical_effect_object = OpticalEffect(self)
        self.check_incomplete = self.segment_retraction_troncated(
            pulling_length)
        if not self.check_incomplete:
            print(
                "\n========================================================================\n")
            print(self.file)
            print(
                "\n========================================================================\n")

            self.calcul_baseline("Press")
            self.calcul_std("Press")

        # self.detected_max_force()
        # #self.check_alignment_curve()

        # self.curve_approach_analyze()
        #self.graphics['graph_approach'] = plt.show()
        #self.graphics['graph_retraction'] = plt.show()

    ################################################################################################

    def __str__(self):
        """
        Determines what is displayed when the object is printed

        :return:
            self.bead: used ball
            self.cell: cell number
        """
        return self.output['bead'] + " " + self.output['cell']

    ###############################################################################################

    def delta_t(self):
        """
        calculation of the delta t on the whole curve

        :return:
            delta_t: float
                time variation on the whole curve
        """
        delta_t = 0
        for segment in self.dict_segments.values():
            delta_t += segment.real_time()
        return delta_t

    ###############################################################################################

    def identification_main_axis(self):
        """
        Determination of the main axis of the manipulation

        :return:
            main_axis: str
                direction related to phi angle with an interval of -pi +/- epsilon
        """
        scanner = self.parameters_header['header_global']['settings.segment.0.xy-scanner.scanner']
        angle = float(
            self.parameters_header['header_global']['settings.segment.0.direction.phi'])
        epsilon = 0.01
        self.features['main_axis'] = {}
        main_axis = ""
        # la convention pour +/- est : ou est la cellule par rapport a la bille
        # EXPLICITER LA CONVENTION ICI
        if scanner == 'sample-scanner':
            if (-np.pi-epsilon) < angle < (-np.pi+epsilon):
                main_axis = "-x"
            elif (-epsilon) < angle < (epsilon):
                main_axis = "+x"
            elif (-np.pi/2.-epsilon) < angle < (-np.pi/2.+epsilon):
                main_axis = "-y"
            elif (np.pi/2.-epsilon) < angle < (np.pi/2.+epsilon):
                main_axis = "+y"
            else:
                main_axis = "Angle error"
        else:
            main_axis = "Error on the scanner"
        if len(main_axis) == 2:
            self.features["main_axis"]['axe'] = main_axis[1]
            self.features["main_axis"]['sign'] = main_axis[0]
        else:
            print(main_axis)

    ################################################################################################
    def analyzed_curve(self, methods, manual_correction):
        """
        launch of the important steps of the analysis of the characteristic elements for a curve

        :parameters:
            methods:dictionary with all the parameters provided by the user for the analysis
            correction: change from the initial correction mode requested
        """
        optical_state = "No_correction"
        type_curve = self.compare_baseline_start_end(methods['factor_noise'])
        if not manual_correction:
            if methods['optical'] == "Correction":
                try:
                    self.correction_optical_effect_object.automatic_correction(
                        methods['factor_noise'])
                    optical_state = "Auto_correction"
                except:
                    print('index error No correction')
        self.curve_approach_analyze(
            methods['model'].lower(), methods['eta'], methods['bead_radius'], methods['factor_noise'])
        type_curve = self.curve_return_analyze(
            methods['jump_force'], methods['jump_point'], methods['jump_distance'], methods['factor_noise'], type_curve)
        self.features['drug'] = methods['drug']
        self.features['condition'] = methods['condition']
        self.features['tolerance'] = methods['factor_noise']
        self.features['optical_state'] = optical_state
        if manual_correction:
            self.features['type'] = type_curve
        else:
            self.features['automatic_type'] = type_curve
        print(self.features['automatic_type'])
        #print(self.features['type'])

    ################################################################################################

    def calcul_baseline(self, name_segment, corrected_data=False, axe="", range_data=1000):
        """
        Determination of the baseline of the curve by calculating
        the average over the first or the last 200 points

        :parameters:
            name_segment: str
                name of the segment on which to search the baseline
        :return:
            basseline: float
                value of the baseline as a function of the segment
        """

        baseline = 0
        if axe == "":
            axe = self.features["main_axis"]['axe']
        segment = self.dict_segments[name_segment]
        if segment.header_segment['segment-settings.style'] == "motion":
            if corrected_data:
                data_analyze = segment.corrected_data
            else:
                data_analyze = segment.data
            if name_segment == "Press":
                baseline = data_analyze[axe + 'Signal1'][0:range_data].mean()
                self.features['baseline_press'] = format(
                    baseline * 1e12, '.3E')
            elif name_segment == "Pull":
                baseline = data_analyze[axe + 'Signal1'][-range_data:].mean()
        self.message += "\n" + str(baseline)
        return baseline

    ###############################################################################################

    def calcul_std(self, name_segment, corrected_data=False, axe="", range_data=200):
        """
        Determination of the standard deviation of the curve by calculating
        over the first or the last 200 points
        :parameters:
            name_segment: str
                name of the segment on which to search the baseline
            range_data: int
                nb point for the calcul
        :return:
            std: float
                value of the standard deviation as a function of the segment
        """
        std = 0
        if axe == "":
            axe = self.features["main_axis"]['axe']
        segment = self.dict_segments[name_segment]
        if segment.header_segment['segment-settings.style'] == "motion":
            if corrected_data:
                data_analyze = segment.corrected_data
            else:
                data_analyze = segment.data
            if name_segment == "Press":
                std = data_analyze[axe + 'Signal1'][0:range_data].std()
                self.features['std_press'] = format(std, '.3E')
            elif name_segment == "Pull":
                std = data_analyze[axe + 'Signal1'][-range_data:].std()

        return std

    ###############################################################################################

    def normalization_data(self):
        """
        Normalize the data on the main axis
        """
        # print("normalization_data")
        num_segment = 0
        time_start = 0.0
        choice_axe = ['x', 'y', 'z']
        main_axis = self.features["main_axis"]['axe']
        for axe in choice_axe:
            baseline = self.calcul_baseline("Press", False, axe)
            column = axe + 'Signal1'
            for segment in self.dict_segments.values():
                data = segment.data[[column]].copy()
                data = data.sub(baseline, axis=0)
                segment.corrected_data[column] = data[column] * 1e12

        for segment in self.dict_segments.values():
            segment.corrected_data['time'] = segment.data['time'] - \
                segment.data['time'][0]
            if num_segment == 0 and float(segment.header_segment['segment-settings.duration']) > 0.0:
                time_start = segment.data['time'][0]
            segment.corrected_data['seriesTime'] = segment.data['seriesTime'].sub(
                time_start)
            num_segment += 1
            stiffness = float(
                self.parameters_header['calibrations'][main_axis + 'Signal1_stiffness'].replace(" N/m", ""))
            self.features['stiffness (N/m)'] = format(stiffness, '.3E')
            stiffness = stiffness * (1e12/1e9)
            distances = segment.data['distance'] * 1e9
            forces = segment.corrected_data[column]
            data_corrected_stiffness = distances - forces / stiffness
            segment.corrected_data['distance'] = np.abs(
                data_corrected_stiffness - data_corrected_stiffness[0])  # (nm)
        if self.features["main_axis"]["sign"] == "+":
            self.curve_reversal()

    ###############################################################################################

    def curve_reversal(self):
        """
        According to the main axis transforms all values into their inverse to return the curve
        """
        # print("curve_reversal")
        for segment in self.dict_segments.values():
            segment.corrected_data['xSignal1'] = - \
                segment.corrected_data['xSignal1']
            segment.corrected_data['ySignal1'] = - \
                segment.corrected_data['ySignal1']
            segment.corrected_data['zSignal1'] = - \
                segment.corrected_data['zSignal1']
            # if self.features["main_axis"]['axe'] == 'x':
            #     segment.corrected_data['ySignal1'] = -segment.corrected_data['ySignal1']
            # else:
            #     segment.corrected_data['xSignal1'] = -segment.corrected_data['xSignal1']
            # segment.corrected_data['zSignal1'] = -segment.corrected_data['zSignal1']

    ###############################################################################################

    def retrieve_data_curve(self, type_data="data_original"):
        """
        Visualization of the curves on the three axes with Matplotlib

        :parameters:
            type_data: str
                Determined the dataframe to be used for visualization
                (original data or corrected data)
            y: str
                data on the axe Y for the plot
        :return:
            plot show
        """
        data_total = pd.DataFrame()
        for segment in self.dict_segments.values():
            if type_data == "data_original":
                data_total = pd.concat(
                    [data_total, segment.data], ignore_index=True, verify_integrity=True)
            else:
                data_total = pd.concat(
                    [data_total, segment.corrected_data], ignore_index=True,  verify_integrity=True)
        return data_total

    ##############################################################################################

    def check_alignment_curve(self, threshold_align):
        """
        Determination of the curve well aligned on the main axis

        :parameters:
            force: float
                application force on the cell (Default: 10e-12)
        :return:
            check: bool
                return True if the curve is not well aligned
        """
        # print("check_alignment_curve")
        force_min = 0
        force_min_exp = self.detected_min_force()
        force_min_exp = abs(force_min_exp * (threshold_align/100))
        segment = self.dict_segments["Press"]
        force_threshold = threshold_align * \
            float(
                segment.header_segment['segment-settings.setpoint.value']) / 100
        force_threshold = force_threshold * 1e12
        if force_threshold < force_min_exp:
            force_min = force_min_exp
        else:
            force_min = force_threshold
        self.graphics['threshold alignement'] = force_min
        dict_align = {}
        for segment in self.dict_segments.values():
            dict_align[segment.name] = segment.check_alignment(
                self.features["main_axis"]['axe'], force_min)
        dict_align_final = {}
        for key, value in dict_align.items():
            if 'AL' not in dict_align_final:
                dict_align_final['AL'] = value['AL']
                dict_align_final['axe'] = value['axe']
            else:
                if dict_align_final['AL'] != value['AL']:
                    if dict_align_final['AL'] == 'Yes':
                        dict_align_final['AL'] = value['AL']
                if dict_align_final['axe'] == 'NaN':
                    dict_align_final['axe'] = value['axe']
                else:
                    if len(dict_align_final['axe']) < len(value['axe']):
                        dict_align_final['axe'] = value['axe']
        return dict_align_final

    ##############################################################################################

    def compare_baseline_start_end(self, tolerance):
        """
        Comparison of the beginning and the end of the curve to know
        if there is a break of adhesion and return to normal

        :return:
            :check: bool
                returns true if start and end similarly
        """
        print("compare_baseline")
        type_curve = ""
        baseline_start = self.calcul_baseline("Press", True)
        line_end = self.calcul_baseline("Pull", True)
        std_start = self.calcul_std("Press", True)
        # print('baseline_start: ', baseline_start)
        # print('std_start: ', std_start)
        # print('line_end: ', line_end)
        # print(line_end)
        # print(baseline_start - std_start*tolerance)
        # print(baseline_start + std_start*tolerance)
        check = False
        if (baseline_start - std_start*tolerance) < line_end < (baseline_start + std_start*tolerance):
            self.message += "\nbaseline_end Ok\n"
            print("baseline_end Ok")
            check = True
            type_curve = None
            #self.features["automatic_type"] = None
        elif(baseline_start + std_start*tolerance) < line_end:
            self.message += "\nBaseline_end NO\n"
            print("Baseline_end NO")
            type_curve = "ITU"
            #self.features["automatic_type"] = "ITU"
        else:
            self.message += "\nBaseline_end NO\n"
            print("Baseline_end NO")
            type_curve = "RE"
            #self.features["automatic_type"] = "RE"

        return type_curve

    #############################################################################################

    def detected_min_force(self, tolerance=10):
        """
        Determination on the approach segment of the ball,
        if the contact between the ball and the cell has been correctly executed

        :parameters:
            tolerance: int
                pourcentage de tolérance pour la force expérimentale

        :return:
            contact: bool
                True if the contact is within the maximum force range
            data_max: float
                maximum value of the contact segment
        """
        print("detected_max_force: ")
        main_axis = self.features["main_axis"]['axe']
        data_min_curve = 0
        index_data_min_curve = 0
        time_min_curve = 0
        index_data_max_curve = 0
        force_max_curve = 0
        for segment in self.dict_segments.values():
            data = segment.corrected_data[main_axis + 'Signal1']
            time = segment.corrected_data['seriesTime']
            if data_min_curve > data[data.argmin()]:
                time_min_curve = time[data.argmin()]
                data_min_curve = data[data.argmin()]
                index_data_min_curve = data.argmin()
            if force_max_curve < data[data.argmax()]:
                force_max_curve = data[data.argmax()]
                index_data_max_curve = data.argmax()
        data_min_approach = 0
        index_data_min = 0
        segment = self.dict_segments["Press"]
        force = float(
            segment.header_segment["segment-settings.setpoint.value"])
        force = -force * 1e12
        interval = force * tolerance / 100
        data = segment.corrected_data[main_axis + 'Signal1']
        data_min_approach = min(data)
        if math.isclose(force, data_min_approach, rel_tol=tolerance):
            index_data_min = data.argmin()
        else:
            for index_data in range(0, len(data), 1):
                if (force - interval) > data[index_data] > (force + interval):
                    if data[index_data] < data_min_approach:
                        data_min_approach = data[index_data]
                        index_data_min = index_data
        self.features["force_min_press"] = {
            'index': index_data_min, 'value': data_min_approach}
        self.features['force_min_curve'] = {
            'index': index_data_min_curve, 'value': data_min_curve}
        self.features['time_min_curve'] = {
            'index': index_data_min_curve, 'value': time_min_curve}
        self.features['force_max_curve'] = {
            'index': index_data_max_curve, 'value': force_max_curve}
        return data_min_curve

    #############################################################################################

    def fit_model_approach(self, data_corrected_stiffness, contact_point, k, baseline):
        """
        Determination contact model between the bead and the cell

        :parameters:
            model: str
                model
        """
        fit = np.array(data_corrected_stiffness)
        fit[fit < contact_point] = baseline
        if self.features['model'] == "linear":
            fit[fit > contact_point] = k * \
                (fit[fit > contact_point]-contact_point) + baseline
        elif self.features['model'] == "sphere":
            fit[fit > contact_point] = k * \
                (fit[fit > contact_point]-contact_point)**(3/2) + baseline
        return fit

    #############################################################################################
    def fit_curve_approach(self, tolerance):
        """
        creation of the data fit for the curve

        :parameters:
            tolerance: noise threshold in number of times the standard deviation

        :return:
            f_parameters: fit parameters describing the model
            fitted: force data corresponding to the model 
        """
        print('fit_curve_approach')
        main_axis = self.features["main_axis"]['axe']
        segment = self.dict_segments["Press"]
        force_data = segment.corrected_data[main_axis + 'Signal1']
        distance_data = np.abs(segment.corrected_data['distance'])
        time_data = segment.corrected_data['seriesTime']
        #self.graphics['y_smooth_Press'] = self.smooth(force_data, time_data, 151, 2)
        index_contact, line_pos_threshold = Curve.retrieve_contact(
            force_data, "Press", tolerance)
        self.graphics['threshold_press'] = line_pos_threshold
        baseline = force_data[0:300].mean() + force_data[0:300].std()  # y0
        x_1 = distance_data[len(distance_data)-index_contact]
        y_1 = force_data[len(force_data)-index_contact]
        x_2 = distance_data[len(distance_data)-10]
        y_2 = force_data[len(force_data)-10]
        k = (y_2 - y_1) / (x_2 - x_1)
        contact_point = distance_data[index_contact-1]  # x0
        self.features['contact_point'] = {
            'index': index_contact, 'value': force_data[index_contact]}
        #initial_guesses_accuracy = [10**(9), 10**3, 1]
        initial_guesses_accuracy = [contact_point, k, baseline]
        f_parameters = curve_fit(
            self.fit_model_approach, distance_data, force_data, initial_guesses_accuracy)
        #self.message += str(f_parameters)
        fitted = self.fit_model_approach(
            distance_data, f_parameters[0][0], f_parameters[0][1], f_parameters[0][2])
        self.graphics['fitted_Press'] = fitted

        return f_parameters, fitted

    ##################################################################################################

    @staticmethod
    def retrieve_contact(data_analyze, segment, tolerance):
        """
        Allows to determine the contact point of the ball with the cell and contact release cell

        """
        print('retrieve_contact')
        list_index_contact = []
        index_contact = 0
        line_pos_threshold = ""
        if segment == "Press":
            baseline = data_analyze[0:200].mean()
            std = data_analyze[0:200].std()
            line_pos_threshold = np.full(len(data_analyze), std*tolerance)
        else:
            baseline = data_analyze[len(data_analyze)-300:].mean()
            std = data_analyze[len(data_analyze)-300:].std()
            line_pos_threshold = np.full(len(data_analyze), std*tolerance)
        for index in range(len(data_analyze)-1, -1, -1):
            if baseline - std < data_analyze[index] < abs(baseline) + abs(std):
                list_index_contact.append(index)
        if segment == "Press":
            index_contact = list_index_contact[0]
        else:
            index_contact = list_index_contact[-1]
        print(index_contact)
        return index_contact, line_pos_threshold

    ###################################################################################################

    def curve_approach_analyze(self, model, eta, bead_ray, tolerance):
        """
        TODO
        """
        self.features['model'] = model
        error = None
        young = None
        error_young = None
        #error_contact = None
        slope = None
        f_parameters, fitted = self.fit_curve_approach(tolerance)
        if np.isfinite(np.sum(f_parameters[1])):
            error = np.sqrt(np.diag(f_parameters[1]))
        if self.features['model'] == 'linear':
            slope = f_parameters[0][1]
            self.features['slope (pN/nm)'] = format(slope, '.2E')
            if isinstance(error, np.ndarray):
                error_young = error[1]
                self.features['error (pN/nm)'] = format(error_young, '.2E')
            else:
                self.features['error (pN/nm)'] = error_young
            self.message += "Slope (pN/nm) = " + \
                str(slope) + " +/-" + str(error_young)
            print("Slope (pN/nm) = " + str(slope) +
                    " +/-" + str(error_young))
        elif self.features['model'] == 'sphere':
            young = Curve.determine_young_modulus(
                f_parameters[0][1], eta, bead_ray)
            error_young = Curve.determine_young_modulus(
                error[1], eta, bead_ray)
            if young.any() < 0.0:
                young = None
                error_young = None
            self.features['young (Pa)'] = young
            self.features['error young (Pa)'] = error_young
            self.message += "Young modulus (Pa) = " + \
                str(young) + " +/-" + str(error_young)
            print("Young modulus (Pa) = " +
                    str(young) + " +/- " + str(error_young))
        else:
            self.message += "Model error"
            self.message += "Trace not processed"
            print("Model error")
            print("Trace not processed")

        length_segment = float(
            self.dict_segments['Press'].header_segment['segment-settings.length'])
        time_segment = float(
            self.dict_segments['Press'].header_segment['segment-settings.duration'])
        vitesse = round(length_segment/time_segment, 1)
        self.dict_segments['Press'].features['vitesse'] = vitesse

    ###################################################################################################

    @staticmethod
    def fit_model_retraction(data, k, point_release, endline):
        """
        TODO
        """
        fit = np.array(data)
        fit[fit < point_release] = k * \
            (fit[fit < point_release]-point_release) + endline
        fit[fit >= point_release] = endline
        return fit

    ##################################################################################################

    def fit_curve_retraction(self, seuil_jump_force, seuil_nb_point, seuil_jump_distance, tolerance, type_curve):
        """
        TODO
        """
        main_axis = self.features["main_axis"]['axe']
        segment = self.dict_segments['Pull']
        force_data = segment.corrected_data[main_axis + 'Signal1']
        distance_data = np.abs(segment.corrected_data['distance'])
        time_data = segment.corrected_data['time']
        mean_final_points = force_data[len(force_data)-300:].mean()
        std_final_points = force_data[len(force_data)-200:].std()
        #line_pos_threshold = np.full(len(force_data),std_final_points*tolerance)
        endline = mean_final_points - std_final_points  # y0
        y_smooth = self.smooth(force_data, time_data, 151, 2)
        index_release, line_pos_threshold = Curve.retrieve_contact(
            force_data, "Pull", tolerance)
        self.graphics['threshold_pull'] = line_pos_threshold
        self.features['point_release'] = {
            'index': index_release, 'value': force_data[index_release]}
        if index_release > 40:
            x_1 = distance_data[index_release-20]
            y_1 = force_data[index_release-20]
        else:
            x_1 = distance_data[20]
            y_1 = force_data[20]
        if index_release > 100:
            x_2 = distance_data[index_release-100]
            y_2 = force_data[index_release-100]
        else:
            x_2 = distance_data[0]
            y_2 = force_data[0]
        k = (y_1 - y_2) / (x_1 - x_2)
        point_release = distance_data[index_release]  # x0
        index_return_end_line = Curve.retrieve_retour_line_end(
            y_smooth, mean_final_points, std_final_points, tolerance)
        # initial_guess = [10**(9), point_release, 10**3]
        initial_guesses_accuracy = [k, point_release, endline]
        f_parameters, f_covariance = curve_fit(
            Curve.fit_model_retraction, distance_data, force_data, initial_guesses_accuracy)
        self.message += str(f_parameters)
        self.features['Pente (pN/nm)'] = f_parameters[1]
        fitted = Curve.fit_model_retraction(
            distance_data, f_parameters[0], f_parameters[1], f_parameters[2])
        self.graphics['fitted_Pull'] = fitted
        self.graphics['y_smooth_Pull'] = y_smooth
        self.features['point_return_endline'] = {
            'index': 'NaN', 'value': 'NaN'}
        self.features['point_transition'] = {
            'index': 'NaN', 'value (pN)': 'NaN'}
        if type_curve == None:
            index_force_max = y_smooth[index_release:index_release+1500].argmax()
            if y_smooth[index_force_max] <= seuil_jump_force:
                type_curve = 'NAD'
            else:
                if index_return_end_line is not None:
                    self.features['point_return_endline'] = {
                        'index': index_return_end_line, 'value': y_smooth[index_return_end_line]}
                    index_transition = index_return_end_line - 10
                        # int(segment.header_segment['segment-settings.num-points'])//1000
                    self.features['point_transition'] = {
                        'index': index_transition, 'value (pN)': y_smooth[index_transition]}
                    index_force_max = force_data[index_release:index_return_end_line].argmax(
                    )
                    jump_force_start_pull = force_data[index_force_max] - \
                        force_data[index_release]
                    jump_distance_start_pull = distance_data[index_force_max] - \
                        distance_data[index_release]
                    jump_nb_points = index_return_end_line - index_force_max
                    jump_distance_end_pull = distance_data[index_release] - \
                        distance_data[index_return_end_line]
                    jump_force_end_pull = force_data[index_release] - \
                        force_data[index_return_end_line]
                    self.features['jump_force_start_pull (pN)'] = jump_force_start_pull
                    self.features['jump_distance_start_pull (nm)'] = jump_distance_start_pull
                    self.features['jump_distance_end_pull (nm)'] = jump_distance_end_pull
                    self.features['jump_force_end_pull (pN)'] = jump_force_end_pull
                    if jump_nb_points < seuil_nb_point and jump_distance_end_pull < seuil_jump_distance:
                        type_curve = 'AD'
                    else:
                        type_curve = 'FTU'
        else:
            index_force_max = force_data.argmax()
        self.features['force_max_pull'] = {
            'index': index_force_max, 'value': force_data[index_force_max]}

        return type_curve

    ####################################################################################################################################

    def derivation(self, force_data, time_data, n):
        derivation = []
        for index in range(n//2, len(force_data)-n//2, 1):
            derivation.append((force_data[index+n//2] - force_data[index-n//2])/(
                time_data[index+n//2] - time_data[index-n//2]))
            #derivation.append((distance_data[index+n//2]- distance_data[index-n//2])/(n * (time_data[index+1] - time_data[index])))
        derivation = np.array(derivation)
        return derivation

    #####################################################################################################################################
    def curve_return_analyze(self, seuil_jump_force, seuil_nb_point, seuil_jump_distance, tolerance, type_curve):
        """
        TODO
        """
        main_axis = self.features["main_axis"]['axe']
        #segment_go = self.dict_segments["Press"]
        #optical_effect = Curve.retrieve_contact(segment_go.corrected_data[main_axis + 'Signal1'], 'optical')
        segment_return = self.dict_segments["Pull"]
        force_data = segment_return.corrected_data[main_axis + 'Signal1']
        distance_data = np.abs(
            segment_return.corrected_data['distance'] - segment_return.corrected_data['distance'][0])
        mean = force_data[len(force_data)-len(force_data)//3:].mean()
        std = force_data[len(force_data)-len(force_data)//3:].std()
        type_curve = self.fit_curve_retraction(
            seuil_jump_force, seuil_nb_point, seuil_jump_distance, tolerance, type_curve)

        return type_curve
    #####################################################################################################################################

    @staticmethod
    def retrieve_retour_line_end(data_analyze, mean_final_points, std_final_points, nb_std):
        """
        TODO
        """
        threshold = np.abs(mean_final_points + std_final_points * nb_std)
        threshold2 = np.abs(mean_final_points + std_final_points * 8)
        # print("threshold: ", threshold)
        # print("threshold2: ", threshold2)
        list_data_return_endline = []
        index_return_endline = None
        data_analyze = np.array(data_analyze)
        data_analyze_reverse = np.flip(data_analyze)
        list_data_return_endline = data_analyze_reverse[(
            data_analyze_reverse > threshold) & (data_analyze_reverse < threshold2)]
        if list_data_return_endline.size != 0:
            index_return_endline = np.where(
                data_analyze == list_data_return_endline[0])[0][0]
            #index_return_endline = len(data_analyze) - index_return_endline
        return index_return_endline

    ######################################################################################################################################

    @staticmethod
    def determine_young_modulus(k, eta, bead_ray):
        """
        Young modulus calculation

        :parameters:
            k: float
                directing coefficient of the slope
            eta: float
                compressibility constant
            bead_ray: float
                size of the ball radius

        :return:
            young:float
                young module
        """
        indentation_depth = 10  # la profondeur d'indentation (m)
        # eta = ratio de Poisson (adimensionnel)
        young = np.around(abs(k) * 1e6 * 3/4 * (1 - eta**2) /
                          np.sqrt(bead_ray * indentation_depth**3), 2)  # radius in nm
        return young

    ######################################################################################################################################

    def segment_retraction_troncated(self, pulling_length):
        """
        Checking the length of the last segment

        :parameters:
            pulling_length: int
                Percentage of length to accept the curve 

        """
        # segment = self.dict_segments['Press']
        # nb_point_segment = segment.header_segment['.num-points']
        # print(nb_point_segment)
        segment = self.dict_segments['Pull']
        nb_point_segment = int(
            segment.header_segment['segment-settings.num-points'])
        size_data = len(
            segment.data[self.features["main_axis"]['axe'] + 'Signal1'])
        check_segment_troncated = True
        if nb_point_segment == size_data:
            check_segment_troncated = False
        else:
            if nb_point_segment * int(pulling_length)/100 <= size_data:
                check_segment_troncated = False
        return check_segment_troncated

    ##################################################################################################################
    @staticmethod
    def test_fit(time_data, slope, offset):
        return slope*time_data + offset

    ######################################################################################################################################

    # @staticmethod

    def smooth(self, force_data, time_data, window_length=51, order_polynome=3):
        """
        Allows to reduce the noise on the whole curve

        :parameters:
            force_data: Series
                values of y 
            window_length: int (odd)
                size of the sliding window

        :return:
            values of y smoothing
        """

        # print("smooth")
        # tck, u = splprep([time_data[::50], force_data[::50]])
        # print('tck: ', tck)
        # print('u: ', u)
        # y_smooth = splev(u, tck, der=0)
        # print(len(y_smooth[1]))
        # derive = splev(u, tck, der=1)
        # print('derive: ', len(derive[1]))
        #derive = spalde(distance_data, tck)
        # inds = time_data.argsort()
        # force_data_order = force_data[inds]
        # index_max_pull = force_data.argmax()
        # if index_max_pull < len(force_data)-1000:
        #     # print(time_data[index_max_pull:])
        #     # print(force_data[index_max_pull:])
        #     spl = UnivariateSpline(
        #         time_data[index_max_pull:], force_data[index_max_pull:], s=2049, ext=1)
        # #spl.set_smoothing_factor(200)
        # derive = spl.derivative(1)
        # derive_second = spl.derivative(2)
        # #fig, ax1 = plt.subplots()

        # # ax1.plot(time_data, force_data)
        # # ax1.plot(time_data,spl(time_data), 'r-')

        # # ax1.plot(time_data, np.zeros(len(force_data)), color='black')
        # # ax2 = ax1.twinx()
        # #ax2.plot(time_data, derive(time_data), color='green')
        # # ax2.plot(time_data, derive_second(time_data), color='green')
        # index_derive_max = derive(time_data).argmin()
        # index_min_derive_second = derive_second(time_data).argmin()
        # index_max_derive_second = derive_second(time_data).argmax()
        # index_min_derive_second = index_min_derive_second

        # self.graphics['milieu_pente'] = index_derive_max
        # self.graphics['index_max_derive'] = index_max_derive_second
        # self.graphics['min_derive'] = index_min_derive_second
        #index_derive_max = np.where(derive(time_data) == derive_max)[0]
        # derive_neg = np.negative(derive_second(time_data))
        # derive_neg_max = derive_neg.max()
        # index_derive_neg_max = np.where(derive_neg == derive_neg_max)[0]-18
        #peaks, _ = find_peaks(derive(time_data), height=100000, prominence=1)
        # peaks = find_peaks(derive_second(time_data), prominence=20000000)
        # if len(peaks[0]) == 1:
        #     print('peaks: ', peaks[0])
        # else:
        #     print(peaks[0])

        # peak = find_peaks_cwt(derive_second)
        # print(peak)

        # ax1.plot(time_data[index_derive_max], force_data[index_derive_max], "D", color='orange')
        # ax1.plot(time_data[index_min_derive_second], force_data[index_min_derive_second], "D", color='pink')
        # ax1.plot(time_data[index_max_derive_second], force_data[index_max_derive_second], "D", color='purple')

        #ax1.plot(time_data[index_derive_neg_max[0]], force_data[index_derive_neg_max], "o")
        # plt.plot(derive[0], derive[1])
        #plt.plot(time_data, force_data, linewidth=0, marker='x')

        # plt.show()

        # bspline = BSpline()
        # print(bspline)
        # print(dir(bspline))
        y_smooth = savgol_filter(force_data, window_length, order_polynome)

        return y_smooth

    ######################################################################################################################################

    def add_feature(self, add_key, add_value):
        """
        Adding features to our "features" dictionary

        :parameters:
            add_key:str
                name of the key to add
            add_value: str, int, dict, list
                data structure to add as value to the key
        """
        self.features[add_key] = add_value

    ######################################################################################################################################

    def creation_output_curve(self):
        for key_features, value_features in self.features.items():
            if isinstance(value_features, dict):
                for key_dict, value_dict in value_features.items():
                    if key_dict == key_features or key_features.split('_')[-1] == key_dict:
                        self.output[key_features] = value_dict
                    else:
                        self.output[key_features + "_" + key_dict] = value_dict
            else:
                self.output[key_features] = value_features
        date = self.file.split('-')[1]
        hour = self.file.split('-')[2].split('.')[0:4]
        hour = '.'.join(hour)
        self.output['Date'] = date
        self.output['Hour'] = hour
        self.output['theorical_contact_force (N)'] = format(float(
            self.parameters_header['header_global']['settings.segment.0.setpoint.value']), '.1E')
        time_segment_pause = 0
        for segment in self.dict_segments.values():
            if segment.header_segment['segment-settings.style'] == 'pause':
                time_segment_pause = float(
                    segment.header_segment['segment-settings.duration'])
                self.output['time_segment_pause_' +
                            segment.name + ' (s)'] = time_segment_pause
            elif segment.header_segment['segment-settings.style'] == 'motion':
                length = segment.header_segment['segment-settings.length']
                self.output['theorical_distance_' + segment.name +
                            ' (m)'] = format(float(length), '.1E')
                freq = format(float(segment.header_segment['segment-settings.num-points'])/float(
                    segment.header_segment['segment-settings.duration']), '.1E')
                self.output['theorical_freq_' + segment.name + ' (Hz)'] = freq
                speed = format(float(segment.header_segment['segment-settings.length'])/float(
                    segment.header_segment['segment-settings.duration']), '.1E')
                self.output['theorical_speed_' +
                            segment.name + ' (m/s)'] = speed
