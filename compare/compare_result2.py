import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


class Compare:
    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2
        self.pre_processing()
        self.compare_type_courbe()

    def pre_processing(self):
        self.df1 = self.df1.set_index(self.df1['Unnamed: 0'])
        self.df2 = self.df2.set_index(self.df2['Filename'])
        self.df1 = self.df1.sort_index(axis=0)
        self.df2 = self.df2.sort_index(axis=0)
        
        #preprocessing df1
        self.df1.drop(self.df1[self.df1['automatic_type'] == 'INC'].index, inplace=True)
        self.df1['slope (pN/nm)'] = abs(self.df1['slope (pN/nm)'])
        self.df1['stiffness (pN/nm)'] = self.df1['stiffness (N/m)'] * 1e3
        self.df1['theorical_contact_force (pN)'] = self.df1['theorical_contact_force (N)'] * 1e12
        self.df1.drop(['Unnamed: 0', 'model', 'treat_supervised', 'drug', 'tolerance', 'theorical_distance_Press (m)', 
        'theorical_speed_Press (m/s)', 'theorical_freq_Press (Hz)', 'theorical_distance_Pull (m)', 'theorical_speed_Pull (m/s)', 
        'theorical_freq_Pull (Hz)', 'baseline_press (pN)', 'contact_point_index', 'contact_point_value  (pN)', 'force_min_press_index', 
        'force_min_press_value (pN)', 'point_release_index', 'force_max_pull_index', 'point_return_endline_index', 'point_release_value (pN)',
        'AL', 'AL_axe', 'point_transition_index', 'point_transition_value (pN)', 'Pente (pN/nm)', 'jump_force_start_pull (pN)',
        'jump_distance_start_pull (nm)', 'jump_force_end_pull (pN)', 'Date', 'Hour', 'bead', 'cell', 'valid_fit_press', 'valid_fit_pull', 
        'theorical_contact_force (N)', 'stiffness (N/m)', 'condition'], axis=1, inplace=True)
        self.df1 = self.df1.fillna(0)

        #preprocessing df2
        self.df2.drop(['Filename', 'Unnamed: 0', 'Aire au min (pN*nm)', 'Aire au jump (pN*nm)', 
                'Position Min Force (nm)','Retrace Fitting frame (#pts)', 'Direction',
                'Couple', 'Distance (µm)', 'Speed (µm/s)', "Trace's fit convergence", "Retrace's fit convergence", 
                'Bead', 'Cell', 'Condition'], axis=1, inplace=True)

        self.df2.rename(columns={'Type of event':'automatic_type', 'Type of event (corrected)':'type',
                'Constant (pN/nm)': 'stiffness (pN/nm)', 'Force contact (pN)':'theorical_contact_force (pN)',
                'Slope (pN/nm)': 'slope (pN/nm)', '[error]':'error (pN/nm)',
                'SD baseline retrace (pN)':'std_press (pN)','Sens':'main_axis', 
                'Time break (s)':'time_segment_pause_Wait1 (s)', 'Min Force (pN)':'force_max_pull_value (pN)', 
                'Jump force (pN)': 'point_return_endline_value (pN)',
                'Jump end (nm)':'jump_distance_end_pull (nm)'}, inplace=True)
        self.df2 = self.df2.fillna(0)
        
        self.df2.loc[self.df2['Tube cassé ?'] == 'yes', 'automatic_type'] = 'FTU'
        self.df2.loc[self.df2['Tube cassé ?'] == 'yes', 'type'] = 'FTU'
        self.df2.loc[self.df2['Tube cassé ?'] == 'no', 'automatic_type'] = 'ITU'
        self.df2.loc[self.df2['Tube cassé ?'] == 'no', 'type'] = 'ITU'
        self.df2['automatic_type'] = self.df2['automatic_type'].str.upper()
        self.df2['type'] = self.df2['type'].str.upper()
        self.df2['main_axis'] = self.df2['main_axis'].str.lower()
        self.df2.drop(['Tube cassé ?'], axis=1, inplace=True)
        self.df2.sort_index(axis=0, ascending=True)
        
        list_label = list(self.df1.columns)
        self.df2= self.df2[list_label]
        print('df_moi: \n', self.df1['automatic_type'])
        print('df_khalil: \n', self.df2['automatic_type'])

    def compare_type_courbe(self):  
        list_type_no_same = [] 
        for index, row in self.df1.iterrows():
            if row['automatic_type'] != self.df2.loc[index]['automatic_type']:
                list_type_no_same.append(index)
        print(list_type_no_same)

    def compare_nb_curve(self, directory):
        dir = Path(directory)
        list_files = list(dir.glob('b[1-9]c[1-9]*.txt'))
        nb_files = len(list_files)
        check_nb_files_df1 = False
        check_nb_files_df2 = False
        if nb_files == len(self.df1.index):
            check_nb_files_df1 = True
        if nb_files == len(self.df2.index):
            check_nb_files_df2 = True
        return check_nb_files_df1, check_nb_files_df2

    def compare_main_axis(self):
        list_check_main_axis = []
        for index, row in self.df1.iterrows():
            if row['main_axis'] != self.df2.loc[index]['main_axis']:
                print(row['main_axis'], self.df2.loc[index]['main_axis'])
                list_check_main_axis.append(True)
        print(list_check_main_axis)
    
    def percentage_type(self):
        nb_type_file1 = self.df1['automatic_type'].value_counts()
        nb_type_file2 = self.df2['automatic_type'].value_counts()
        nb_curve_1 = len(self.df1.index)
        nb_curve_2 = len(self.df2.index)
        print(nb_type_file1)
        print(nb_type_file2)
        percentage_type = {'df_Thierry': {}, 'df_Khalil': {}}
        for index, nb in nb_type_file1.items():
            percentage_type['df_Thierry'][index] = nb/nb_curve_1 * 100
        for index, nb in nb_type_file2.items():
            percentage_type['df_Khalil'][index] = nb/nb_curve_2 * 100
        print(percentage_type)

    def scatter_plot_header(self):
        columns = ['stiffness (pN/nm)', 'theorical_contact_force (pN)', 'time_segment_pause_Wait1 (s)']
        df1 = self.df1[columns]
        df2 = self.df2[columns]
        fig = plt.figure()
        pos_graph = 1
        for col in df1.columns:
            ax = plt.subplot(130 + pos_graph)
            ax.scatter(df1[col], df2[col], label=col, alpha=0.25)
            x0=np.linspace(df1[col].min(),df1[col].max(),100)
            ax.plot(x0,x0, color='k', alpha=0.25)
            ax.set_title(col)
            ax.set_xlabel('Output Thierry')
            ax.set_ylabel('Output Khalil')
            pos_graph += 1
        plt.subplots_adjust(wspace = 0.3, hspace = 0.2)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()
    
    def plot_characteristics_measure(self):
        columns = ['slope (pN/nm)', 'error (pN/nm)', 'std_press (pN)', 'force_max_pull_value (pN)',
                    'point_return_endline_value (pN)', 'jump_distance_end_pull (nm)']
        df1 = self.df1[columns]
        df2 = self.df2[columns]
        fig = plt.figure()
        pos_graph = 1
        nb_line_graph = 0
        for col in df1.columns:
            ax=""
            if len(columns) % 3 == 0:
                nb_line_graph = len(columns)//3
                ax = plt.subplot(int(str(nb_line_graph) + '3' + str(pos_graph)))
            else:
                nb_line_graph = len(columns)//2
                ax = plt.subplot(int(str(nb_line_graph) + '4' + str(pos_graph)))
            for index, row in df1.iterrows():
                ax.scatter(df1.loc[index][col], df2.loc[index][col], label=index)
                x0 = np.linspace(df1[col].min(),df1[col].max(),100)
                ax.plot(x0,x0, color='k', alpha=0.25)
                ax.set_title(col)
                ax.set_xlabel('Output Thierry')
                ax.set_ylabel('Output Khalil')
                ax.set_xlim()
            handle = ax.get_legend_handles_labels()
            pos_graph += 1
        plt.subplots_adjust(wspace = 0.5, hspace = 0.8)
        fig.legend(handles=handle[0], labels=handle[1])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()
        


if __name__ == "__main__":
    df_moi = pd.read_csv('Result/output_2022-03-21_17-49-41.csv', sep='\t')
    df_khalil = pd.read_csv('Result/20220314-150113-results.csv', sep='\t')
    compare = Compare(df_moi, df_khalil)
    check1, check2 = compare.compare_nb_curve('./data_test/txt')
    print(check1, check2)
    compare.compare_main_axis()
    compare.percentage_type()
    compare.scatter_plot_header()
    compare.plot_characteristics_measure()
