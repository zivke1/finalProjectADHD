
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer
import csv
import os
from pathlib import Path
# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, ttk
import tkinter
from tkinter import filedialog as fd
from matplotlib.figure import Figure
import build_load_dataset.gui as load_dataset_win
import build_generate_graphs.gui as generate_graphs_win
from utilities.json_creator import OutputHandler as jh
from networkx.readwrite import json_graph
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import networkx as nx
import functools
import subprocess

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

control_group_graphsList = []
treatment_group_graphsList = []
def read_graphs(fileNameADHD,fileNameControl):
    control_group_graphsList.clear()
    treatment_group_graphsList.clear()
    jsonH = jh()
    #ZK change to name from the previous window
    graph_data = jsonH.read_json("DB2\graphs", fileNameControl, "Graphs")
    for g in graph_data:
        H = json_graph.adjacency_graph(g)
        control_group_graphsList.append(H)

    graph_data = jsonH.read_json("DB2\graphs", fileNameADHD, "Graphs")
    for g in graph_data:
        H = json_graph.adjacency_graph(g)
        treatment_group_graphsList.append(H)

def graph_feature_average_degree_Press(parent = None):
    frequencyBand = parent.FB.get()
    if frequencyBand == 'Choose frequency band':
        parent.children['label_asterisk'].config(text="Choose frequency band first")
        return
    else:
        parent.children['label_asterisk'].config(text="")
        parent.children['label_finish'].config(text="")
    read_graphs(parent.name_of_ADHD_graph_file+frequencyBand.lower()+parent.precentageOfThisTest,parent.name_of_control_graph_file+frequencyBand.lower()+parent.precentageOfThisTest)
    treatment_average_degreeList = []
    control_average_degreeList = []
    for patientTreatment in treatment_group_graphsList:
        treatment_average_degreeList.append(sum(dict(patientTreatment.degree()).values()) / len(patientTreatment))
    for patientControl in control_group_graphsList:
        control_average_degreeList.append(sum(dict(patientControl.degree()).values()) / len(patientControl))

    show_graph(treatment_average_degreeList, control_average_degreeList, frequencyBand, 'Average degree', parent)

def graph_feature_Transitivity_Press(parent = None):
    frequencyBand = parent.FB.get()
    if frequencyBand == 'Choose frequency band':
        parent.children['label_asterisk'].config(text="Choose frequency band first")
        return
    else:
        parent.children['label_asterisk'].config(text="")
        parent.children['label_finish'].config(text="")
    read_graphs(parent.name_of_ADHD_graph_file + frequencyBand.lower() + parent.precentageOfThisTest,
                parent.name_of_control_graph_file + frequencyBand.lower() + parent.precentageOfThisTest)
    treatmentTransitivityList = []
    controlTransitivityList = []
    for patientADHD in treatment_group_graphsList:
        treatmentTransitivityList.append(nx.transitivity(patientADHD))

    for patientcontrol in control_group_graphsList:
        controlTransitivityList.append(nx.transitivity(patientcontrol))

    show_graph(treatmentTransitivityList, controlTransitivityList, frequencyBand, 'Transitivity', parent)

def graph_feature_AvgClust_Press(parent = None):
    frequencyBand = parent.FB.get()
    if frequencyBand == 'Choose frequency band':
        parent.children['label_asterisk'].config(text="Choose frequency band first")
        return
    else:
        parent.children['label_asterisk'].config(text="")
        parent.children['label_finish'].config(text="")
    read_graphs(parent.name_of_ADHD_graph_file + frequencyBand.lower() + parent.precentageOfThisTest,
                parent.name_of_control_graph_file + frequencyBand.lower() + parent.precentageOfThisTest)
    treatmentAverageClusteringList = []
    controlAverageClusteringList = []
    for patientTreatment in treatment_group_graphsList:
        treatmentAverageClusteringList.append(nx.average_clustering(patientTreatment))

    for patientcontrol in control_group_graphsList:
        controlAverageClusteringList.append(nx.average_clustering(patientcontrol))

    show_graph(treatmentAverageClusteringList, controlAverageClusteringList, frequencyBand, 'Average Clustering', parent)

def graph_feature_density_Press(parent = None):
    frequencyBand = parent.FB.get()
    if frequencyBand == 'Choose frequency band':
        parent.children['label_asterisk'].config(text="Choose frequency band first")
        return
    else:
        parent.children['label_asterisk'].config(text="")
        parent.children['label_finish'].config(text="")
    read_graphs(parent.name_of_ADHD_graph_file + frequencyBand.lower() + parent.precentageOfThisTest,
                parent.name_of_control_graph_file + frequencyBand.lower() + parent.precentageOfThisTest)
    treatmentDensityList = []
    controldensityList = []
    for patientTreatment in treatment_group_graphsList:
        treatmentDensityList.append(nx.density(patientTreatment))

    for patientcontrol in control_group_graphsList:
        controldensityList.append(nx.density(patientcontrol))

    show_graph(treatmentDensityList, controldensityList, frequencyBand, 'density', parent)

def graph_feature_degAC_Press(parent = None):
    frequencyBand = parent.FB.get()
    if frequencyBand == 'Choose frequency band':
        parent.children['label_asterisk'].config(text="Choose frequency band first")
        return
    else:
        parent.children['label_asterisk'].config(text="")
        parent.children['label_finish'].config(text="")
    read_graphs(parent.name_of_ADHD_graph_file + frequencyBand.lower() + parent.precentageOfThisTest,
                parent.name_of_control_graph_file + frequencyBand.lower() + parent.precentageOfThisTest)
    treatmentDegreeAssortativityCoefficientList = []
    controlDegreeAssortativityCoefficientList = []
    for patientTreatment in treatment_group_graphsList:
        treatmentDegreeAssortativityCoefficientList.append(nx.degree_assortativity_coefficient(patientTreatment))

    for patientcontrol in control_group_graphsList:
        controlDegreeAssortativityCoefficientList.append(nx.degree_assortativity_coefficient(patientcontrol))

    show_graph(treatmentDegreeAssortativityCoefficientList, controlDegreeAssortativityCoefficientList, frequencyBand, 'Degree Assortativity Coefficient', parent)

def graph_feature_global_efficiency_Press(parent = None):#we can take this also
    frequencyBand = parent.FB.get()
    if frequencyBand == 'Choose frequency band':
        parent.children['label_asterisk'].config(text="Choose frequency band first")
        return
    else:
        parent.children['label_asterisk'].config(text="")
        parent.children['label_finish'].config(text="")
    read_graphs(parent.name_of_ADHD_graph_file+frequencyBand.lower()+parent.precentageOfThisTest,parent.name_of_control_graph_file+frequencyBand.lower()+parent.precentageOfThisTest)
    treatment_global_efficiencyList = []
    control_global_efficiencyList = []
    for patientTreatment in treatment_group_graphsList:
        G2 = patientTreatment.to_undirected()
        treatment_global_efficiencyList.append(nx.global_efficiency(G2))

        # treatmentDegree_pearson_correlation_coefficientList.append(nx.degree_pearson_correlation_coefficient(patientTreatment))

    for patientcontrol in control_group_graphsList:
        G2 = patientcontrol.to_undirected()
        control_global_efficiencyList.append(nx.global_efficiency(G2))

    show_graph(treatment_global_efficiencyList, control_global_efficiencyList, frequencyBand,'Global Efficiency', parent)


# def graph_feature_average_shortest_path_length_Press(parent = None):#we can take this also
#     # nx.(G)
#     frequencyBand = parent.FB.get()
#     if frequencyBand == 'Choose frequency band':
#         parent.children['label_asterisk'].config(text="Choose frequency band first")
#         return
#     else:
#         parent.children['label_asterisk'].config(text="")
#     read_graphs(parent.name_of_ADHD_graph_file+frequencyBand.lower()+parent.precentageOfThisTest,parent.name_of_control_graph_file+frequencyBand.lower()+parent.precentageOfThisTest)
#     treatment_average_degreeList = []
#     controlDegree_pearson_correlation_coefficientList = []
#     for patientTreatment in treatment_group_graphsList:
#         G2 = patientTreatment.to_undirected()
#         treatment_average_degreeList.append(nx.k_nearest_neighbors(patientTreatment))
#
#         # treatmentDegree_pearson_correlation_coefficientList.append(nx.degree_pearson_correlation_coefficient(patientTreatment))
#
#     for patientcontrol in control_group_graphsList:
#         # G2 = patientcontrol.to_undirected()
#         controlDegree_pearson_correlation_coefficientList.append(nx.k_nearest_neighbors(patientcontrol))
#
#     fig = Figure(figsize=(5, 5), dpi=100)
#     data = [treatment_average_degreeList, controlDegree_pearson_correlation_coefficientList]
#     plot1 = fig.add_subplot(111)
#     bp = plot1.boxplot(data, patch_artist=True,
#                        notch='True')
#
#     colors = ['#0000FF', '#00FF00']
#
#     for patch, color in zip(bp['boxes'], colors):
#         patch.set_facecolor(color)
#         # patch.set(title = "11")
#     fig.suptitle(frequencyBand+' Global efficiency', fontsize=12, fontweight='bold')
#     plot1.set_xlabel('Treatment                              Control')
#     canvas = FigureCanvasTkAgg(fig,
#                                master=parent)
#     canvas.get_tk_widget().place(x=570, y=110)

def show_graph(ADHDFeatureList, controlFeatureList, frequencyBand, feature_name, parent=None):
    fig = Figure(figsize=(5, 5), dpi=100)
    data = [ADHDFeatureList, controlFeatureList]
    plot1 = fig.add_subplot(111)
    bp = plot1.boxplot(data, patch_artist=True, notch='True')
    colors = ['#0000FF', '#00FF00']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        # patch.set(title = "11")

    fig.suptitle(frequencyBand + ' ' + feature_name, fontsize=12, fontweight='bold')
    plot1.set_xlabel('Treatment                                  Control')

    canvas = FigureCanvasTkAgg(fig,master=parent)
    canvas.get_tk_widget().place(x=570, y=110)

def exportBtn(freqToListOfGraphs_ADHD_group_individuals, freqToListOfGraphs_control_group_individuals, parent=None):
    filePathName = fd.askdirectory()
    parent.children['label_finish'].config(text="")
    if filePathName == "":
        parent.children['label_asterisk'].config(text="Please choose a folder for export")
        return
    parent.children['label_asterisk'].config(text="")
    export_data_btn(freqToListOfGraphs_ADHD_group_individuals, filePathName +'\ADHD_group_features_individuals.csv')
    export_data_btn(freqToListOfGraphs_control_group_individuals, filePathName +'\Control_group_features_individuals.csv')
    parent.children['label_finish'].config(text="Finish to export")
    os.startfile(filePathName)

def export_data_btn(freqToListOfGraphs_group_individuals, file_name):
    with open(file_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        fieldnames = ['Name', 'Alpha', 'Beta', 'Gamma', 'Theta', 'Delta']
        # write the header
        writer.writerow(fieldnames)

    for p in freqToListOfGraphs_group_individuals:
        ##### error -- this is not good, continue from here
        alpha_degree_average_degree = sum(dict(freqToListOfGraphs_group_individuals[p]['alphaList'][0].degree()).values()) / len(freqToListOfGraphs_group_individuals[p]['alphaList'][0])
        alpha_density = nx.density(freqToListOfGraphs_group_individuals[p]['alphaList'][0])
        alpha_average_clustering = nx.average_clustering(freqToListOfGraphs_group_individuals[p]['alphaList'][0])
        alpha_transitivity = nx.transitivity(freqToListOfGraphs_group_individuals[p]['alphaList'][0])
        alpha_global_efficiency = nx.global_efficiency(freqToListOfGraphs_group_individuals[p]['alphaList'][0].to_undirected())

        beta_degree_average_degree = sum(dict(freqToListOfGraphs_group_individuals[p]['betaList'][0].degree()).values()) / len(freqToListOfGraphs_group_individuals[p]['betaList'][0])
        beta_density = nx.density(freqToListOfGraphs_group_individuals[p]['betaList'][0])
        beta_average_clustering = nx.average_clustering(freqToListOfGraphs_group_individuals[p]['betaList'][0])
        beta_transitivity = nx.transitivity(freqToListOfGraphs_group_individuals[p]['betaList'][0])
        beta_global_efficiency = nx.global_efficiency(freqToListOfGraphs_group_individuals[p]['betaList'][0].to_undirected())

        gamma_degree_average_degree = sum(dict(freqToListOfGraphs_group_individuals[p]['gammaList'][0].degree()).values()) / len(freqToListOfGraphs_group_individuals[p]['gammaList'][0])
        gamma_density = nx.density(freqToListOfGraphs_group_individuals[p]['gammaList'][0])
        gamma_average_clustering = nx.average_clustering(freqToListOfGraphs_group_individuals[p]['gammaList'][0])
        gamma_transitivity = nx.transitivity(freqToListOfGraphs_group_individuals[p]['gammaList'][0])
        gamma_global_efficiency = nx.global_efficiency(freqToListOfGraphs_group_individuals[p]['gammaList'][0].to_undirected())

        delta_degree_average_degree = sum(dict(freqToListOfGraphs_group_individuals[p]['deltaList'][0].degree()).values()) / len(freqToListOfGraphs_group_individuals[p]['deltaList'][0])
        delta_density = nx.density(freqToListOfGraphs_group_individuals[p]['deltaList'][0])
        delta_average_clustering = nx.average_clustering(freqToListOfGraphs_group_individuals[p]['deltaList'][0])
        delta_transitivity = nx.transitivity(freqToListOfGraphs_group_individuals[p]['deltaList'][0])
        delta_global_efficiency = nx.global_efficiency(freqToListOfGraphs_group_individuals[p]['deltaList'][0].to_undirected())

        theta_degree_average_degree = sum(dict(freqToListOfGraphs_group_individuals[p]['thetaList'][0].degree()).values()) / len(freqToListOfGraphs_group_individuals[p]['thetaList'][0])
        theta_density = nx.density(freqToListOfGraphs_group_individuals[p]['thetaList'][0])
        theta_average_clustering = nx.average_clustering(freqToListOfGraphs_group_individuals[p]['thetaList'][0])
        theta_transitivity = nx.transitivity(freqToListOfGraphs_group_individuals[p]['thetaList'][0])
        theta_global_efficiency = nx.global_efficiency(freqToListOfGraphs_group_individuals[p]['thetaList'][0].to_undirected())

        data = [
            [p],
            ['Nodes', freqToListOfGraphs_group_individuals[p]['alphaList'][0].number_of_nodes(), freqToListOfGraphs_group_individuals[p]['betaList'][0].number_of_nodes(), freqToListOfGraphs_group_individuals[p]['gammaList'][0].number_of_nodes(), freqToListOfGraphs_group_individuals[p]['thetaList'][0].number_of_nodes(), freqToListOfGraphs_group_individuals[p]['deltaList'][0].number_of_nodes()],
            ['Edges', freqToListOfGraphs_group_individuals[p]['alphaList'][0].number_of_edges(), freqToListOfGraphs_group_individuals[p]['betaList'][0].number_of_edges(), freqToListOfGraphs_group_individuals[p]['gammaList'][0].number_of_edges(), freqToListOfGraphs_group_individuals[p]['thetaList'][0].number_of_edges(), freqToListOfGraphs_group_individuals[p]['deltaList'][0].number_of_edges()],
            ['average degree',       alpha_degree_average_degree,    beta_degree_average_degree,       gamma_degree_average_degree,       theta_degree_average_degree,       delta_degree_average_degree],
            ['density',              alpha_density,                  beta_density,                     gamma_density,                     theta_density,                     delta_density],
            ['average clustering',   alpha_average_clustering,       beta_average_clustering,          gamma_average_clustering,          theta_average_clustering,          delta_average_clustering],
            ['transitivity',         alpha_transitivity,             beta_transitivity,                gamma_transitivity,                theta_transitivity,                delta_transitivity],
            ['global efficiency',    alpha_global_efficiency,        beta_global_efficiency,           gamma_global_efficiency,           theta_global_efficiency,           delta_global_efficiency]
        ]

        write_to_csv(file_name, data)

def write_to_csv(name , data):
    with open(name, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write multiple rows
        writer.writerows(data)

def on_closing(root):
    dir = '..\DB2\graphs'
    # check first if dir exist
    # delete dir content
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    root.destroy()

class win:

    def __init__(self, file_name_ADHD, file_name_control ,precentage, files_name, freqToListOfGraphs_ADHD_group_individuals, freqToListOfGraphs_control_group_individuals):
        window = Tk()
        window.title("Analyze Data")
        window.geometry("1170x687")
        window.configure(bg = "#FFFFFF")

        # those are the names of the json files in DB2 folder that contains the graphs of
        # the ADHD and control group that were generated in the previous window -> generate graphs
        window.name_of_ADHD_graph_file = file_name_ADHD
        window.name_of_control_graph_file = file_name_control
        window.precentageOfThisTest = precentage

        canvas = Canvas(
            window,
            bg = "#FFFFFF",
            height = 687,
            width = 1170,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )

        canvas.place(x = 0, y = 0)
        canvas.create_rectangle(
            0.0,
            0.0,
            234.0,
            684.0,
            fill="#9272EB",
            outline="")


        canvas.create_rectangle(
            234.0,
            0.0,
            1167.0,
            684.0,
            fill="#E2D8EF",
            outline="")

        button_image_1 = PhotoImage(
            file=relative_to_assets("button_1.png"))
        button_1 = Button(
            image=button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: print("button_1 clicked"),
            relief="flat"
        )
        button_1.place(
            x=515.0,
            y=82.0,
            width=600.0,
            height=560.0
        )

        button_image_2 = PhotoImage(
            file=relative_to_assets("button_2.png"))
        button_2 = Button(
            image=button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: {
                window.destroy():
                generate_graphs_win.win()
            },
            relief="flat"
        )
        button_2.place(
            x=34.0,
            y=92.0,
            width=168.0,
            height=37.0
        )

        button_image_3 = PhotoImage(
            file=relative_to_assets("button_3.png"))
        button_3 = Button(
            image=button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: {
                    window.destroy():
                    load_dataset_win.win()
            },
            relief="flat"
        )
        button_3.place(
            x=34.0,
            y=154.0,
            width=168.0,
            height=37.0
        )

        canvas.create_text(
            27.0,
            33.0,
            anchor="nw",
            text="EEG Recordings Analyzer",
            fill="#FFFAFA",
            font=("JejuMyeongjo", 14 * -1)
        )

        button_image_4 = PhotoImage(
            file=relative_to_assets("button_4.png"))
        button_4 = Button(
            image=button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: graph_feature_average_degree_Press(parent=window),
            relief="flat"
        )
        button_4.place(
            x=266.0,
            y=353.0,
            width=178.0,
            height=37.0
        )

        button_image_5 = PhotoImage(
            file=relative_to_assets("button_5.png"))
        button_5 = Button(
            image=button_image_5,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: graph_feature_Transitivity_Press(window),
            relief="flat"
        )
        button_5.place(
            x=266.0,
            y=121.0,
            width=178.0,
            height=37.0
        )

        button_image_6 = PhotoImage(
            file=relative_to_assets("button_6.png"))
        button_6 = Button(
            image=button_image_6,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: graph_feature_AvgClust_Press(window),
            relief="flat"
        )
        button_6.place(
            x=266.0,
            y=179.0,
            width=178.0,
            height=37.0
        )
# export button
        button_image_7 = PhotoImage(
            file=relative_to_assets("button_7.png"))
        button_7 = Button(
            image=button_image_7,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: exportBtn(freqToListOfGraphs_ADHD_group_individuals, freqToListOfGraphs_control_group_individuals, window),
            relief="flat"
        )
        button_7.place(
            x=266.0,
            y=570.0,
            width=178.0,
            height=37.0
        )

        button_image_8 = PhotoImage(
            file=relative_to_assets("button_8.png"))
        button_8 = Button(
            image=button_image_8,
            borderwidth=0,
            highlightthickness=0,
            command=lambda:graph_feature_density_Press(window),
            # command=lambda: graph_feature_average_shortest_path_length_Press(window),
            relief="flat"
        )
        button_8.place(
            x=266.0,
            y=237.0,
            width=178.0,
            height=37.0
        )

        button_image_9 = PhotoImage(
            file=relative_to_assets("button_9.png"))
        button_9 = Button(
            image=button_image_9,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: graph_feature_global_efficiency_Press(window),
            relief="flat"
        )
        button_9.place(
            x=266.0,
            y=295.0,
            width=178.0,
            height=37.0
        )

        window.FB = tkinter.StringVar()
        month_cb = tkinter.ttk.Combobox(name='frequencyBandsCB', textvariable=window.FB)
        month_cb.place(x=270.0,
                       y=70.0,
                       width=168.0,
                       height=34.0)
        month_cb['values'] = ['Alpha', 'Beta', 'Gamma', 'Theta', 'Delta']
        # month_cb['state'] = 'readonly'
        # month_cb.current(0)
        month_cb.insert(0, "Choose frequency band")
        # error_no_frequency_band_was_chosen(canvas)


        title = "Graph features of files: " + files_name
        canvas.create_text(
            280.0,
            25.0,
            anchor="nw",
            text=title,
            fill="#000000",
            font=("JejuMyeongjo", 24 * -1)
        )

        label_asterisk = tkinter.Label(name='label_asterisk', fg="red", bg='#E2D8EF').place(x=270, y=620)
        label_finish = tkinter.Label(name='label_finish', fg="green", bg='#E2D8EF').place(x=270, y=640)
        # window.children['label_asterisk'].config(text="Choose frequency band first")

        window.resizable(False, False)
        on_close_with_params = functools.partial(on_closing, window)
        window.protocol("WM_DELETE_WINDOW", on_close_with_params)
        window.mainloop()
