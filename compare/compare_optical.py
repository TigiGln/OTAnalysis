import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


class CompareOptical:
    def __init__(self, df1, df2, name_df1, name_df2):
        self.df1 = df1
        self.df2 = df2
        self.name_df1 = name_df1
        self.name_df2 = name_df2
        self.df1 = self.df1.set_index(self.df1['Unnamed: 0'])
        self.df2 = self.df2.set_index(self.df1['Unnamed: 0'])
        self.df1.drop(['Unnamed: 0'], axis=1, inplace=True)
        self.df2.drop(['Unnamed: 0'], axis=1, inplace=True)
        self.intersection(self.df1.columns, self.df2.columns)

    def intersection(self, lst1, lst2):
        set_difference = set(lst1).symmetric_difference(set(lst2))
        list_difference = list(set_difference)
        common_columns = list(set(lst1) & set(lst2))
        if len(lst1) > len(lst2):
            self.df1.drop(list_difference, axis=1, inplace=True)
        else:
            self.df2.drop(list_difference, axis=1, inplace=True)
        self.df1 = self.df1[common_columns]
        self.df2 = self.df2[common_columns]

    def compare_df(self):
        df_compare = self.df1.compare(self.df2)
        print(df_compare)
        return df_compare

    def plot_characteristics_measure(self, columns):
        df1 = self.df1[columns]
        df2 = self.df2[columns]
        fig = plt.figure()
        pos_graph = 1
        nb_line_graph = 0
        for col in df1.columns:
            ax = ""
            if len(columns) % 3 == 0:
                nb_line_graph = len(columns)//3
                ax = plt.subplot(
                    int(str(nb_line_graph+1) + '3' + str(pos_graph)))
            elif len(columns) % 4 == 0:
                nb_line_graph = len(columns)//4
                ax = plt.subplot(
                    int(str(nb_line_graph+1) + '4' + str(pos_graph)))
            else:
                nb_line_graph = len(columns)//2
                ax = plt.subplot(
                    int(str(nb_line_graph) + '4' + str(pos_graph)))
            for index, row in df1.iterrows():
                ax.scatter(df1.loc[index][col],
                           df2.loc[index][col], color='grey', alpha=0.5, label=index)
                x0 = np.linspace(df1[col].min(), df1[col].max(), 100)
                ax.plot(x0, x0, color='k', alpha=0.25)
                ax.set_title(col)
                ax.set_xlabel(self.name_df1)
                ax.set_ylabel(self.name_df2)
            handle = ax.get_legend_handles_labels()
            pos_graph += 1
        plt.subplots_adjust(wspace=0.5, hspace=0.8)
        # if len(columns) >= 6:
        #     fig.legend(handles=handle[0], labels=handle[1],
        #                ncol=3, loc='lower center')
        # else:
        #     fig.legend(handles=handle[0], labels=handle[1],
        #                loc='lower right')
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()


if __name__ == "__main__":
    PATH = 'result_test/'
    FILE_NO_CORRECTION = 'output_2022-04-11_14-55-40.csv'
    df_no_correction = pd.read_csv(
        PATH + FILE_NO_CORRECTION, sep='\t')
    FILE_WITH_CORRECTION = 'output_2022-04-11_14-56-13.csv'
    df_with_correction = pd.read_csv(
       PATH + FILE_WITH_CORRECTION , sep='\t')
    FILE_WITH_MANUAL_CORRECTION = 'output_2022-04-11_17-49-36.csv'
    df_with_correction_manual = pd.read_csv(
        PATH + FILE_WITH_MANUAL_CORRECTION, sep='\t')
    compare = CompareOptical(df_no_correction, df_with_correction,
                             'Output No correction', 'output with correction')
    # compare = CompareOptical(df_with_correction_manual, df_with_correction,
    #                          'Output manual correction ', 'output with correction')
    df_compare = compare.compare_df()
    name_file = 'compare/compare_' + FILE_NO_CORRECTION + '_' + FILE_WITH_CORRECTION
    df_compare.to_csv(name_file, sep='\t', encoding='utf-8', na_rep="NaN")

    columns1 = ['baseline_press (pN)', 'std_press (pN)', 'slope (pN/nm)',
                'error (pN/nm)', 'Pente (pN/nm)']
    compare.plot_characteristics_measure(columns1)
    columns2 = ['contact_point_index', 'contact_point_value  (pN)',
                'force_min_press_index', 'force_min_press_value (pN)',
                'point_release_index', 'point_release_value (pN)',
                'force_max_pull_index', 'force_max_pull_value (pN)']
    compare.plot_characteristics_measure(columns2)
    columns3 = ['point_transition_index', 'point_transition_value (pN)',
                'jump_force_start_pull (pN)', 'jump_distance_start_pull (nm)',
                'jump_distance_end_pull (nm)', 'jump_force_end_pull (pN)']
    compare.plot_characteristics_measure(columns3)
    columns4 = ['point_return_endline_index', 'point_return_endline_value (pN)',
                'force_min_curve_index', 'force_min_curve_value', 'time_min_curve_index',
                'time_min_curve_value', 'force_max_curve_index', 'force_max_curve_value']
    compare.plot_characteristics_measure(columns4)
    if compare.df1['optical_state'][0] == "Auto" or compare.df1['optical_state'][0] == "Manual":
        if compare.df2['optical_state'][0] == "Auto" or compare.df2['optical_state'][0] == "Manual":
            columns5 = ['contact_theorical_press_index', 'contact_theorical_pull_index',
                        'contact_theorical_press_value', 'contact_theorical_pull_value']
            compare.plot_characteristics_measure(columns5)
