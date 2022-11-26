
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, ttk
import tkinter
from matplotlib.figure import Figure
import build_load_dataset.gui as load_dataset_win
import build_generate_graphs.gui as generate_graphs_win
from utilities.json_creator import OutputHandler as jh
from networkx.readwrite import json_graph
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import networkx as nx


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

control_group_graphsList = []
ADHD_group_graphsList = []
def read_graphs(fileNameADHD,fileNameControl):
    control_group_graphsList.clear()
    ADHD_group_graphsList.clear()
    jsonH = jh()
    #ZK change to name from the previous window
    graph_data = jsonH.read_json("DB2\graphs", fileNameControl, "Graphs")
    for g in graph_data:
        H = json_graph.adjacency_graph(g)
        control_group_graphsList.append(H)

    graph_data = jsonH.read_json("DB2\graphs", fileNameADHD, "Graphs")
    for g in graph_data:
        H = json_graph.adjacency_graph(g)
        ADHD_group_graphsList.append(H)

def graph_feature_DPCC_Press(parent = None):
    frequencyBand = parent.FB.get()
    if frequencyBand == 'Choose frequency band':
        parent.children['label_asterisk'].config(text="Choose frequency band first")
        return
    else:
        parent.children['label_asterisk'].config(text="")
    read_graphs(parent.name_of_ADHD_graph_file+frequencyBand.lower()+parent.precentageOfThisTest,parent.name_of_control_graph_file+frequencyBand.lower()+parent.precentageOfThisTest)
    ADHDdegree_pearson_correlation_coefficientList = []
    controlDegree_pearson_correlation_coefficientList = []
    for patientADHD in ADHD_group_graphsList:
        ADHDdegree_pearson_correlation_coefficientList.append(nx.degree_pearson_correlation_coefficient(patientADHD))

    for patientcontrol in control_group_graphsList:
        controlDegree_pearson_correlation_coefficientList.append(nx.degree_pearson_correlation_coefficient(patientcontrol))

    fig = Figure(figsize=(5, 5), dpi=100)
    data = [ADHDdegree_pearson_correlation_coefficientList, controlDegree_pearson_correlation_coefficientList]
    plot1 = fig.add_subplot(111)
    bp = plot1.boxplot(data, patch_artist=True,
                       notch='True')

    colors = ['#0000FF', '#00FF00']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        # patch.set(title = "11")

    fig.suptitle(frequencyBand+' Degree Pearson Correlation Coefficient', fontsize=12, fontweight='bold')
    plot1.set_xlabel('ADHD                                  Control')
    canvas = FigureCanvasTkAgg(fig,
                               master=parent)
    canvas.get_tk_widget().place(x=570, y=110)

def graph_feature_Transitivity_Press(parent = None):
    frequencyBand = parent.FB.get()
    if frequencyBand == 'Choose frequency band':
        parent.children['label_asterisk'].config(text="Choose frequency band first")
        return
    else:
        parent.children['label_asterisk'].config(text="")
    read_graphs(parent.name_of_ADHD_graph_file + frequencyBand.lower() + parent.precentageOfThisTest,
                parent.name_of_control_graph_file + frequencyBand.lower() + parent.precentageOfThisTest)
    ADHDtransitivityList = []
    controlTransitivityList = []
    for patientADHD in ADHD_group_graphsList:
        ADHDtransitivityList.append(nx.transitivity(patientADHD))

    for patientcontrol in control_group_graphsList:
        controlTransitivityList.append(nx.transitivity(patientcontrol))

    fig = Figure(figsize=(5, 5), dpi=100)
    data = [ADHDtransitivityList,controlTransitivityList]
    plot1 = fig.add_subplot(111)
    bp = plot1.boxplot(data, patch_artist=True,
                       notch='True')

    colors = ['#0000FF', '#00FF00']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        # patch.set(title = "11")

    fig.suptitle(frequencyBand +' Transitivity', fontsize=12, fontweight='bold')
    plot1.set_xlabel('ADHD                                  Control')
    canvas = FigureCanvasTkAgg(fig,
                               master=parent)
    canvas.get_tk_widget().place(x=570,y=110)
    # canvas.get_tk_widget().grid(row=0, column=0, pady=20, padx=20, sticky="wens")

def graph_feature_AvgClust_Press(parent = None):
    frequencyBand = parent.FB.get()
    if frequencyBand == 'Choose frequency band':
        parent.children['label_asterisk'].config(text="Choose frequency band first")
        return
    else:
        parent.children['label_asterisk'].config(text="")
    read_graphs(parent.name_of_ADHD_graph_file + frequencyBand.lower() + parent.precentageOfThisTest,
                parent.name_of_control_graph_file + frequencyBand.lower() + parent.precentageOfThisTest)
    ADHDAverageClusteringList = []
    controlAverageClusteringList = []
    for patientADHD in ADHD_group_graphsList:
        ADHDAverageClusteringList.append(nx.average_clustering(patientADHD))

    for patientcontrol in control_group_graphsList:
        controlAverageClusteringList.append(nx.average_clustering(patientcontrol))

    fig = Figure(figsize=(5, 5), dpi=100)
    data = [ADHDAverageClusteringList, controlAverageClusteringList]
    plot1 = fig.add_subplot(111)
    bp = plot1.boxplot(data, patch_artist=True,
                       notch='True')

    colors = ['#0000FF', '#00FF00']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        # patch.set(title = "11")

    fig.suptitle(frequencyBand + ' Average Clustering', fontsize=12, fontweight='bold')
    plot1.set_xlabel('ADHD                                  Control')
    canvas = FigureCanvasTkAgg(fig,
                               master=parent)
    canvas.get_tk_widget().place(x=570, y=110)
    # canvas.get_tk_widget().grid(row=0, column=0, pady=20, padx=20, sticky="wens")

def graph_feature_density_Press(parent = None):
    frequencyBand = parent.FB.get()
    if frequencyBand == 'Choose frequency band':
        parent.children['label_asterisk'].config(text="Choose frequency band first")
        return
    else:
        parent.children['label_asterisk'].config(text="")
    read_graphs(parent.name_of_ADHD_graph_file + frequencyBand.lower() + parent.precentageOfThisTest,
                parent.name_of_control_graph_file + frequencyBand.lower() + parent.precentageOfThisTest)
    ADHDdensityList = []
    controldensityList = []
    for patientADHD in ADHD_group_graphsList:
        ADHDdensityList.append(nx.density(patientADHD))

    for patientcontrol in control_group_graphsList:
        controldensityList.append(nx.density(patientcontrol))

    fig = Figure(figsize=(5, 5), dpi=100)
    data = [ADHDdensityList, controldensityList]
    plot1 = fig.add_subplot(111)
    bp = plot1.boxplot(data, patch_artist=True,
                       notch='True')

    colors = ['#0000FF', '#00FF00']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        # patch.set(title = "11")

    fig.suptitle(frequencyBand +' density', fontsize=12, fontweight='bold')
    plot1.set_xlabel('ADHD                                  Control')
    canvas = FigureCanvasTkAgg(fig,
                               master=parent)
    canvas.get_tk_widget().place(x=570, y=110)

def graph_feature_degAC_Press(parent = None):
    frequencyBand = parent.FB.get()
    if frequencyBand == 'Choose frequency band':
        parent.children['label_asterisk'].config(text="Choose frequency band first")
        return
    else:
        parent.children['label_asterisk'].config(text="")
    read_graphs(parent.name_of_ADHD_graph_file + frequencyBand.lower() + parent.precentageOfThisTest,
                parent.name_of_control_graph_file + frequencyBand.lower() + parent.precentageOfThisTest)
    ADHDDegreeAssortativityCoefficientList = []
    controlDegreeAssortativityCoefficientList = []
    for patientADHD in ADHD_group_graphsList:
        ADHDDegreeAssortativityCoefficientList.append(nx.degree_assortativity_coefficient(patientADHD))

    for patientcontrol in control_group_graphsList:
        controlDegreeAssortativityCoefficientList.append(nx.degree_assortativity_coefficient(patientcontrol))

    fig = Figure(figsize=(5, 5), dpi=100)
    data = [ADHDDegreeAssortativityCoefficientList, controlDegreeAssortativityCoefficientList]
    plot1 = fig.add_subplot(111)
    bp = plot1.boxplot(data, patch_artist=True,
                       notch='True')

    colors = ['#0000FF', '#00FF00']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        # patch.set(title = "11")

    fig.suptitle(frequencyBand +' Degree Assortativity Coefficient', fontsize=12, fontweight='bold')
    plot1.set_xlabel('ADHD                                  Control')
    canvas = FigureCanvasTkAgg(fig,
                               master=parent)
    canvas.get_tk_widget().place(x=570, y=110)


class win:

    def __init__(self, file_name_ADHD, file_name_control ,precentage, files_name):
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
            command=lambda: graph_feature_DPCC_Press(parent=window),
            relief="flat"
        )
        button_4.place(
            x=266.0,
            y=363.0,
            width=178.0,
            height=59.0
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
            command=lambda: print("button_7 clicked"),
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
            relief="flat"
        )
        button_8.place(
            x=266.0,
            y=235.0,
            width=178.0,
            height=37.0
        )

        button_image_9 = PhotoImage(
            file=relative_to_assets("button_9.png"))
        button_9 = Button(
            image=button_image_9,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: graph_feature_degAC_Press(window),
            relief="flat"
        )
        button_9.place(
            x=266.0,
            y=292.0,
            width=178.0,
            height=53.0
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

        label_asterisk = tkinter.Label(name='label_asterisk', fg="red", bg='#E2D8EF').place(x=270, y=630)
        # window.children['label_asterisk'].config(text="Choose frequency band first")

        window.resizable(False, False)
        window.mainloop()
