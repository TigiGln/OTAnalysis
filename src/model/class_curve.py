"""
File describing the instance class of the curved objects
"""
import math
import numpy as np
import pandas as pd
import traceback
# import matplotlib.pyplot as plt
# from matplotlib.figure import Figure
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
# from scipy.interpolate import UnivariateSpline
# from lmfit import Model
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
        ######### Attributes #################
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

        ######### methods ###############
        self.identification_main_axis()
        self.normalization_data()
        self.correction_optical_effect_object = OpticalEffect(self)
        self.check_incomplete = self.segment_retraction_troncated(
            pulling_length)
        if not self.check_incomplete:
            baseline_origin = self.calcul_baseline("Press")
            self.features['baseline_origin_press (N)'] = format(
                    baseline_origin * 1e12, '.3E')
            beseline_corrected =  self.calcul_baseline('Press', True)
            self.features['baseline_corrected_press (pN)'] = format(
                    beseline_corrected * 1e12, '.3E')
            std_origin = self.calcul_std("Press")
            self.features['std_origin_press (N)'] = format(std_origin, '.3E')
            std_corrected = self.calcul_std("Press", True)
            self.features['std_corrected_press (pN)'] = format(std_corrected, '.3E')

    ################################################################################################
    """
    Initialization methods of the curves object:
    - identification_main_axis
    - curbe_reversal
    - normalization_data
    - segment_retraction_troncated
    - calcul_baseline
    - calcul_std
    - detected_min_force
    - check_alignment_curve"""
    ################################################################################################

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
            elif name_segment == "Pull":
                baseline = data_analyze[axe + 'Signal1'][-range_data:].mean()
        self.message += "\n" + str(baseline)
        return baseline

    ################################################################################################

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
            if 'distance' in segment.data:
                distances = segment.data['distance'] * 1e9
                forces = segment.corrected_data[column]
                data_corrected_stiffness = distances - forces / stiffness
                segment.corrected_data['distance'] = np.abs(
                    data_corrected_stiffness - data_corrected_stiffness[0])  # (nm)
        if self.features["main_axis"]["sign"] == "+":
            self.curve_reversal()
    
    ###############################################################################################

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
            elif name_segment == "Pull":
                std = data_analyze[axe + 'Signal1'][-range_data:].std()

        return std

    ###################################################################################################

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

    ###################################################################################################

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
                    if isinstance(dict_align_final, list) and len(dict_align_final['axe']) < len(value['axe']):
                        dict_align_final['axe'] = value['axe']
        return dict_align_final

    ###################################################################################################
    
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
        baseline_start = float(self.features['baseline_corrected_press (pN)'])
        line_end = self.calcul_baseline("Pull", True)
        std_start = float(self.features['std_corrected_press (pN)'])
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

    ###############################################################################################
                                    # Methods common to all segments               
    ###############################################################################################
    def smooth(self, force_data, window_length=51, order_polynome=3):
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
        if window_length % 2 == 0:
            window_length += 1
        y_smooth = savgol_filter(force_data, window_length, order_polynome)

        return y_smooth

    ###############################################################################################

    def retrieve_contact(self, data_analyze, segment, tolerance):
        """
        Allows to determine the contact point of the ball with the cell and contact release cell

        """
        print('retrieve_contact')
        list_index_contact = []
        index_contact = 0
        line_pos_threshold = ""
        baseline = float(self.features['baseline_corrected_press (pN)'])
        std = float(self.features['std_corrected_press (pN)'])
        line_pos_threshold = np.full(len(data_analyze), std*tolerance)
        for index in range(len(data_analyze)-1, -1, -1):
            if baseline - std < data_analyze[index] < abs(baseline) + abs(std):
                list_index_contact.append(index)
        if len(list_index_contact) > 0:
            if segment == "Press":
                index_contact = list_index_contact[0]
            else:
                index_contact = list_index_contact[-1]
        
        return index_contact, line_pos_threshold

    ###############################################################################################
                                    # Analysis methods segment Approach
    ###############################################################################################

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

    ###############################################################################################

    def fit_curve_approach(self, tolerance, window_smooth):
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
        y_smooth = self.smooth(force_data, window_smooth, 2)
        # if 'distance' in segment.corrected_data:
        #     distance_data = np.abs(segment.corrected_data['distance'])
        time_data = segment.corrected_data['seriesTime']
        #self.graphics['y_smooth_Press'] = y_smooth
        index_contact, line_pos_threshold = self.retrieve_contact(
            y_smooth, "Press", tolerance)
        self.graphics['threshold_press'] = line_pos_threshold
        baseline = float(self.features['baseline_corrected_press (pN)']) #y0
        x_1 = time_data[len(time_data)-index_contact]
        y_1 = force_data[len(force_data)-index_contact]
        x_2 = time_data[len(time_data)-10]
        y_2 = force_data[len(force_data)-10]
        k = (y_2 - y_1) / (x_2 - x_1)
        contact_point = time_data[index_contact-1]  # x0
        self.features['contact_point'] = {
            'index': index_contact, 'value': force_data[index_contact]}
        #initial_guesses_accuracy = [10**(9), 10**3, 1]
        initial_guesses_accuracy = [contact_point, k, baseline]
        f_parameters = curve_fit(
            self.fit_model_approach, time_data, y_smooth, initial_guesses_accuracy)
        #self.message += str(f_parameters)
        fitted = self.fit_model_approach(
            time_data, f_parameters[0][0], f_parameters[0][1], f_parameters[0][2])
        self.graphics['fitted_Press'] = fitted

        return f_parameters
    
    ###############################################################################################
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

    ###############################################################################################
    def curve_approach_analyze(self, methods):
        """
        TODO
        """
        self.features['model'] = methods['model'].lower()
        error = None
        young = None
        error_young = None
        #error_contact = None
        slope = None
        f_parameters = self.fit_curve_approach(methods['factor_noise'], methods['width_window_smooth'])
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
                f_parameters[0][1], methods['eta'], methods['bead_radius'])
            error_young = Curve.determine_young_modulus(
                error[1], methods['eta'], methods['bead_radius'])
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
    
    ###############################################################################################
                                        # Analysis methods segment Return
    ################################################################################################

    @staticmethod
    def retrieve_retour_line_end(data_analyze, line_pos_threshold):
        """
        TODO
        """
        # threshold = np.abs(mean_final_points + std_final_points * nb_std)
        # threshold2 = np.abs(mean_final_points + std_final_points * 8)
        # print("threshold: ", threshold)
        # print("threshold2: ", threshold2)
        list_data_return_endline = []
        index_return_endline = None
        data_analyze = np.array(data_analyze)
        data_analyze_reverse = np.flip(data_analyze)
        list_data_return_endline = data_analyze_reverse[(
            data_analyze_reverse > line_pos_threshold[0])]
        if list_data_return_endline.size != 0:
            index_return_endline = np.where(
                data_analyze == list_data_return_endline[0])[0][0] - 1

            #index_return_endline = len(data_analyze) - index_return_endline
        return index_return_endline

    ################################################################################################
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


    ################################################################################################
    
    def fit_curve_retraction(self, methods, type_curve):
        """
        TODO
        """
        ###### data ###############
        segment = self.dict_segments['Pull']
        force_data = segment.corrected_data[self.features["main_axis"]['axe'] + 'Signal1']
        y_smooth = self.smooth(force_data, methods['width_window_smooth'], 2)
        self.graphics['y_smooth_Pull'] = y_smooth
        time_data = segment.corrected_data['time']

        ######## calcul release #########
        index_release, line_pos_threshold = self.retrieve_contact(
            y_smooth, "Pull", methods['factor_noise'])
        self.graphics['threshold_pull'] = line_pos_threshold
        self.features['point_release'] = {
            'index': index_release, 'value': force_data[index_release]}

        ################## calcul guess and fit ################
        ###### guess  ########
        if index_release > 40:
            x_1 = time_data[index_release-20]
            y_1 = y_smooth[index_release-20]
        else:
            x_1 = time_data[20]
            y_1 = y_smooth[20]
        if index_release > 100:
            x_2 = time_data[index_release-100]
            y_2 = y_smooth[index_release-100]
        else:
            x_2 = time_data[0]
            y_2 = y_smooth[0]
        k = (y_1 - y_2) / (x_1 - x_2)
        point_release = time_data[index_release]  # x0
        baseline = float(self.features['baseline_corrected_press (pN)'])
        # initial_guess = [10**(9), point_release, 10**3]
        initial_guesses_accuracy = [k, point_release, baseline]
        ######## fit #########
        f_parameters = curve_fit(
            Curve.fit_model_retraction, time_data, y_smooth, initial_guesses_accuracy)
        self.message += str(f_parameters)
        self.features['Pente (pN/nm)'] = f_parameters[0][1]
        fitted = Curve.fit_model_retraction(
            time_data, f_parameters[0][0], f_parameters[0][1], f_parameters[0][2])
        self.graphics['fitted_Pull'] = fitted

        ############## calcul return point and transition ######################

        index_return_end_line = Curve.retrieve_retour_line_end(
            y_smooth, line_pos_threshold)
        if index_return_end_line != None :
            index_transition = index_return_end_line - 10
            self.features['point_return_endline'] = {
                            'index': index_return_end_line, 'value': y_smooth[index_return_end_line]}
            self.features['point_transition'] = {
                        'index': index_transition, 'value (pN)': y_smooth[index_transition]}
        else:
            self.features['point_return_endline'] = {'index': 'NaN', 'value': 'NaN'}
            self.features['point_transition'] = {'index': 'NaN', 'value (pN)': 'NaN'}

        type_curve = self.classification(methods, type_curve)

        return type_curve

    ###########################################################################################################################

    def classification(self, methods, type_curve ):
        ########### data ############
        segment = self.dict_segments['Pull']
        force_data = segment.corrected_data[self.features["main_axis"]['axe'] + 'Signal1']
        distance_data = None
        if 'distance' in segment.corrected_data:
            distance_data = np.abs(segment.corrected_data['distance'])
        time_data = segment.corrected_data['time']

        ############ points characteristics ##################
        index_release = self.features['point_release']['index']
        index_return_end_line = self.features['point_return_endline']['index']

        ############## classification NAD, AD, FTU ##################
        if type_curve == None:
            index_force_max = self.graphics['y_smooth_Pull'][index_release:index_release+1500].argmax()
            if self.features['force_max_curve']['value'] <= methods['jump_force']:
                type_curve = 'NAD'
            else:
                if index_return_end_line is not None:
                    # if index_release < index_return_end_line:
                    #     index_force_max = self.graphics['y_smooth_Pull'][index_release:index_return_end_line].argmax()
                    # elif index_release > index_return_end_line:
                    #     index_force_max = self.graphics['y_smooth_Pull'][index_return_end_line:index_release].argmax()
                    # else:
                    index_force_max = self.features['force_max_curve']['index']
                    ##################### calcul jump ########################
                    jump_force_start_pull = self.features['force_max_curve']['value'] - \
                        self.graphics['y_smooth_Pull'][index_release]
                    jump_nb_points = index_return_end_line - index_force_max
                    jump_force_end_pull = self.graphics['y_smooth_Pull'][index_return_end_line] - self.graphics['y_smooth_Pull'][index_release]
                    self.features['jump_force_start_pull (pN)'] = jump_force_start_pull
                    self.features['jump_force_end_pull (pN)'] = jump_force_end_pull
                    self.features['jump_nb_points'] = jump_nb_points
                    
                    jump_time_start_pull = time_data[index_force_max] - time_data[index_release]
                    jump_time_end_pull = time_data[index_return_end_line] - time_data[index_release]
                    self.features['jump_time_start_pull (s)'] = jump_time_start_pull
                    self.features['jump_time_end_pull (s)'] = jump_time_end_pull
                    
                    jump_distance_start_pull = 0
                    jump_distance_end_pull = 0
                    
                    if distance_data is not None:
                        print('hey')
                        jump_distance_start_pull = distance_data[index_force_max] - \
                            distance_data[index_release]
                        
                        jump_distance_end_pull = distance_data[index_return_end_line] - distance_data[index_release]
                        self.features['jump_distance_start_pull (nm)'] = jump_distance_start_pull
                    else:
                        speed = float(segment.header_segment['segment-settings.length'])/float(segment.header_segment['segment-settings.duration'])
                        jump_distance_end_pull = (speed*1e9) * jump_time_end_pull
                    self.features['jump_distance_end_pull (nm)'] = jump_distance_end_pull

                    ############### determination AD ou FTU ###############
                    if jump_nb_points < methods['jump_point'] and jump_distance_end_pull < methods['jump_distance']:
                        type_curve = 'AD'
                    else:
                        type_curve = 'FTU'
        else:
            index_force_max = force_data.argmax()
        self.features['force_max_pull'] = {
            'index': index_force_max, 'value': force_data[index_force_max]}

        return type_curve

    ################################################################################################
    
    def curve_return_analyze(self, methods, type_curve):
        """
        TODO
        """
        type_curve = self.fit_curve_retraction(methods, type_curve)

        return type_curve

    ################################################################################################
                                 # Launching methods of analysis of the segments of the curve
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
                except Exception as error:
                    print('###########################################')
                    print(())
                    print(type(error).__name__, ':')
                    print(error)
                    print(traceback.format_exc())
                    print('index error No correction')
                    print('###########################################')
        self.curve_approach_analyze(methods)
        type_curve = self.curve_return_analyze(methods, type_curve)
        self.features['drug'] = methods['drug']
        self.features['condition'] = methods['condition']
        self.features['tolerance'] = methods['factor_noise']
        self.features['optical_state'] = optical_state
        if manual_correction:
            self.features['type'] = type_curve
        else:
            self.features['automatic_type'] = type_curve

    ###############################################################################################
                                 # Methods used in the supervision of the interface
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

    ###################################################################################################

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

    ###################################################################################################
                                            # Method of creating the output 
    ##################################################################################################

    def creation_output_curve(self):
        """
        TODO
        """
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

    ##################################################################################################
                                            # Other methods or test methods
    ##################################################################################################

    # def derivation(self, force_data, time_data, n):
    #     derivation = []
    #     for index in range(n//2, len(force_data)-n//2, 1):
    #         derivation.append((force_data[index+n//2] - force_data[index-n//2])/(
    #             time_data[index+n//2] - time_data[index-n//2]))
    #     derivation = np.array(derivation)
    #     return derivation

    ##################################################################################################
    # @staticmethod
    # def linear_fit(time_data, slope, offset):
    #     return slope*time_data + offset

    ################################################################################################

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