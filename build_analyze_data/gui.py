
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from pathlib import Path
import numpy as np
import networkx as nx
from matplotlib.figure import Figure
from networkx.readwrite import json_graph
from utilities.json_creator import OutputHandler as jh
# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

import build_load_dataset.gui as load_dataset_win
import build_generate_graphs.gui as generate_graphs_win

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

control_group_graphsList = []
ADHD_group_graphsList = []
def read_graphs():
    jsonH = jh()
    graph_data = jsonH.read_json("DB2\graphs", "control_group_graphs75", "Graphs")
    for g in graph_data:
        H = json_graph.adjacency_graph(g)
        control_group_graphsList.append(H)

    graph_data = jsonH.read_json("DB2\graphs", "ADHD_group_graphs75", "Graphs")
    for g in graph_data:
        H = json_graph.adjacency_graph(g)
        ADHD_group_graphsList.append(H)

def graph_feature5Press(parent = None):
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

    fig.suptitle('Degree Pearson Correlation Coefficient', fontsize=14, fontweight='bold')
    plot1.set_xlabel('ADHD                                  Control')
    canvas = FigureCanvasTkAgg(fig,
                               master=parent)
    canvas.get_tk_widget().place(x=570, y=100)


def graph_feature2Press(parent = None):
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

    fig.suptitle('Average Clustering', fontsize=14, fontweight='bold')
    plot1.set_xlabel('ADHD                                  Control')
    canvas = FigureCanvasTkAgg(fig,
                               master=parent)
    canvas.get_tk_widget().place(x=570, y=100)
    # canvas.get_tk_widget().grid(row=0, column=0, pady=20, padx=20, sticky="wens")

def graph_feature4Press(parent = None):
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

    fig.suptitle('Degree Assortativity Coefficient', fontsize=14, fontweight='bold')
    plot1.set_xlabel('ADHD                                  Control')
    canvas = FigureCanvasTkAgg(fig,
                               master=parent)
    canvas.get_tk_widget().place(x=570, y=100)

def graph_feature3Press(parent = None):
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

    fig.suptitle('density', fontsize=14, fontweight='bold')
    plot1.set_xlabel('ADHD                                  Control')
    canvas = FigureCanvasTkAgg(fig,
                               master=parent)
    canvas.get_tk_widget().place(x=570, y=100)

def graph_feature1Press(parent = None):
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

    fig.suptitle('Transitivity', fontsize=14, fontweight='bold')
    plot1.set_xlabel('ADHD                                  Control')
    canvas = FigureCanvasTkAgg(fig,
                               master=parent)
    canvas.get_tk_widget().place(x=570,y=100)
    # canvas.get_tk_widget().grid(row=0, column=0, pady=20, padx=20, sticky="wens")

class win:
    def __init__(self, file_name_ADHD, file_name_control):
        window = Tk()
        window.geometry("1170x687")
        window.configure(bg = "#FFFFFF")

        # those are the names of the json files in DB2 folder that contains the graphs of
        # the ADHD and control group that were generated in the previous window -> generate graphs
        name_of_ADHD_graph_file = file_name_ADHD
        name_of_control_graph_file = file_name_control

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
            y=44.0,
            width=600.0,
            height=600.0
        )

        button_image_2 = PhotoImage(
            file=relative_to_assets("button_2.png"))
        button_2 = Button(
            image=button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda:{
                window.destroy():
                generate_graphs_win.win()
            },
            relief="flat"
        )
        button_2.place(
            x=34.0,
            y=154.0,
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
            y=92.0,
            width=168.0,
            height=37.0
        )

        button_image_4 = PhotoImage(
            file=relative_to_assets("button_4.png"))
        button_4 = Button(
            image=button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: print("button_4 clicked"),
            relief="flat"
        )
        button_4.place(
            x=34.0,
            y=215.0,
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

        button_image_5 = PhotoImage(
            file=relative_to_assets("button_5.png"))
        button_5 = Button(
            image=button_image_5,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: graph_feature5Press(parent=window),
            relief="flat"
        )
        button_5.place(
            x=266.0,
            y=350.0,
            width=178.0,
            height=37.0
        )

        button_image_6 = PhotoImage(
            file=relative_to_assets("button_6.png"))
        button_6 = Button(
            image=button_image_6,
            borderwidth=0,
            highlightthickness=0,
            command=lambda:graph_feature1Press(parent = window),
            relief="flat"
        )
        button_6.place(
            x=266.0,
            y=121.0,
            width=178.0,
            height=37.0
        )

        button_image_7 = PhotoImage(
            file=relative_to_assets("button_7.png"))
        button_7 = Button(
            image=button_image_7,
            borderwidth=0,
            highlightthickness=0,
            command=lambda:graph_feature2Press(parent = window),
            relief="flat"
        )
        button_7.place(
            x=266.0,
            y=179.0,
            width=178.0,
            height=37.0
        )

        button_image_8 = PhotoImage(
            file=relative_to_assets("button_8.png"))
        button_8 = Button(
            image=button_image_8,
            borderwidth=0,
            highlightthickness=0,
            command=lambda:graph_feature3Press(parent = window),
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
            command=lambda: graph_feature4Press(parent=window),
            relief="flat"
        )
        button_9.place(
            x=266.0,
            y=292.0,
            width=178.0,
            height=37.0
        )

        # canvas = Canvas(figsize=(5, 5),
        #        dpi=100)
        # canvas.place(x=266.0,
        #     y=292.0)

        read_graphs()   # read graphs

        window.resizable(False, False)
        window.mainloop()

