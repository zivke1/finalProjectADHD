
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
import os
import tkinter
import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
import build_load_dataset.gui as load_dataset_win
import build_analyze_data.gui as analyze_data_win
from utilities.json_creator import OutputHandler as jh



OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def remove_values_from_matrix_under_precentages(data , precentage):
    # this code is remove all the values under the threshold
    for patient in data:
        # p = patient
        for frequencyBand,matrixes  in patient.items():
            if frequencyBand == 'patient_name':
                continue
            ssr_based_F_testMat = matrixes['ssr_based_F_testMat']
            threshold = np.percentile(ssr_based_F_testMat, int(precentage))
            for indexD1, listD1 in enumerate(ssr_based_F_testMat):
                for indexD2, item in enumerate(listD1):
                    if item < threshold:
                        ssr_based_F_testMat[indexD1][indexD2] = 0



def gen_graphs_pressed(parent = None):
    precentage = parent.children['precentageEntry'].get()
    jsonH = jh()
    if precentage == '':
        parent.children['labelFolderExists'].config(text = "You must enter a threshold value")
        return

    parent.children['labelFolderExists'].config(text="")
    #need to check if it don't select nothing
    diractory = parent.children['listBoxOfDataSet'].selection_get()
    folderPath = ".\\..\\DB\\"+diractory+"\\conclusionMatrixADHD.json"
    f = open(folderPath)
    conclusionMatrixADHD = json.load(f)
    dataADHD = conclusionMatrixADHD['Patients']
    remove_values_from_matrix_under_precentages(dataADHD, precentage)

    freqToListOfGraphs_ADHD_group = {}



    create_graphs(dataADHD, 'ssr_based_F_testMat', freqToListOfGraphs_ADHD_group)  # create graphs for ADHD patients and insert to the list
    jsonH.listOf_graphs_to_json(freqToListOfGraphs_ADHD_group['alphaList'], "ADHD_group_graphs_alpha"+precentage, "DB2\graphs")
    jsonH.listOf_graphs_to_json(freqToListOfGraphs_ADHD_group['betaList'], "ADHD_group_graphs_beta" + precentage,
                                "DB2\graphs")
    jsonH.listOf_graphs_to_json(freqToListOfGraphs_ADHD_group['gammaList'], "ADHD_group_graphs_gamma" + precentage,
                                "DB2\graphs")
    jsonH.listOf_graphs_to_json(freqToListOfGraphs_ADHD_group['thetaList'], "ADHD_group_graphs_theta" + precentage,
                                "DB2\graphs")
    jsonH.listOf_graphs_to_json(freqToListOfGraphs_ADHD_group['deltaList'], "ADHD_group_graphs_delta" + precentage,
                                "DB2\graphs")

    #### above ADHD ; below control  ####

    folderPath = ".\\..\\DB\\"+diractory+"\\conclusionMatrixControl.json"
    f = open(folderPath)
    conclusionMatrixControl = json.load(f)
    dataControl = conclusionMatrixControl['Patients']
    remove_values_from_matrix_under_precentages(dataControl ,precentage)

    freqToListOfGraphs_control_group = {}

    create_graphs(dataControl, 'ssr_based_F_testMat', freqToListOfGraphs_control_group)  # create graphs for control patients and insert to the list
    # jsonH.listOf_graphs_to_json(freqToListOfGraphs_control_group, "control_group_graphs"+precentage, "DB2\graphs")
    jsonH.listOf_graphs_to_json(freqToListOfGraphs_control_group['alphaList'], "control_group_graphs_alpha"+precentage, "DB2\graphs")
    jsonH.listOf_graphs_to_json(freqToListOfGraphs_control_group['betaList'], "control_group_graphs_beta" + precentage,
                                "DB2\graphs")
    jsonH.listOf_graphs_to_json(freqToListOfGraphs_control_group['gammaList'], "control_group_graphs_gamma" + precentage,
                                "DB2\graphs")
    jsonH.listOf_graphs_to_json(freqToListOfGraphs_control_group['thetaList'], "control_group_graphs_theta" + precentage,
                                "DB2\graphs")
    jsonH.listOf_graphs_to_json(freqToListOfGraphs_control_group['deltaList'], "control_group_graphs_delta" + precentage,
                                "DB2\graphs")
 # destroy window at the end of the function
    name_of_ADHD_graph_file = "ADHD_group_graphs_"
    name_of_control_graph_file = "control_group_graphs_"
    parent.destroy()
    analyze_data_win.win(name_of_ADHD_graph_file, name_of_control_graph_file,precentage)

def create_graphs(patients_data, matrix_name,freqToListOfGraphs):
    ## create graph from ssr_based_F_testMat

    freqToListOfGraphs['alphaList'] = []
    freqToListOfGraphs['betaList'] = []
    freqToListOfGraphs['gammaList'] = []
    freqToListOfGraphs['thetaList'] = []
    freqToListOfGraphs['deltaList'] = []
    for patient in patients_data:
        for frequencyBand, matrixes in patient.items():
            if frequencyBand == 'patient_name':
                continue

            ssr_based_F_testMat = matrixes[matrix_name]
            # for i in range(len(ssr_based_F_testMat)):
            #     for j in range(len(ssr_based_F_testMat[i])):
            #         if ssr_based_F_testMat[i][j] is None:
            #             ssr_based_F_testMat[i][j] = 0
            listOflist_to_npArray = np.array([np.array(i) for i in ssr_based_F_testMat])

            G = nx.from_numpy_matrix(listOflist_to_npArray, create_using=nx.DiGraph)
            if matrixes['frequancy_band_name'] == 'alphaList':
                freqToListOfGraphs['alphaList'].append(G)
            elif matrixes['frequancy_band_name'] == 'betaList':
                freqToListOfGraphs['betaList'].append(G)
            elif matrixes['frequancy_band_name'] == 'gammaList':
                freqToListOfGraphs['gammaList'].append(G)
            elif matrixes['frequancy_band_name'] == 'thetaList':
                freqToListOfGraphs['thetaList'].append(G)
            elif matrixes['frequancy_band_name'] == 'deltaList':
                freqToListOfGraphs['deltaList'].append(G)

## Global value
showInfoText = True

def present_info_text(parent = None):       # information Icon msg - each click show / remove msg
    global showInfoText
    if showInfoText:
        parent.children['labelInfo'].config(text = "Threshold value will determine the percentage of data we will discard.")
        showInfoText = not showInfoText
        return
    parent.children['labelInfo'].config(text="")   ## <--- TODO: find better way to remove label
    showInfoText = not showInfoText



class win:
    def __init__(self, *args, **kwargs):
        window = Tk()
        window.title("Generate Graphs")
        window.geometry("1170x687")
        window.configure(bg = "#FFFFFF")


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
            x=34.0,
            y=92.0,
            width=168.0,
            height=37.0
        )

        button_image_2 = PhotoImage(
            file=relative_to_assets("button_2.png"))
        button_2 = Button(
            image=button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: {
                    window.destroy():
                    load_dataset_win.win()
            },
            relief="flat"
        )
        button_2.place(
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

        canvas.create_rectangle(
            234.0,
            0.0,
            1167.0,
            684.0,
            fill="#E2D8EF",
            outline="")

        button_image_3 = PhotoImage(
            file=relative_to_assets("button_3.png"))
        button_3 = Button(
            image=button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: gen_graphs_pressed(parent = window),
            relief="flat"
        )
        button_3.place(
            x=817.0,
            y=489.0,
            width=206.0,
            height=50.0
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
            x=400.0,
            y=127.0,
            width=600.0,
            height=300.0
        )

        entry_image_1 = PhotoImage(
            file=relative_to_assets("entry_1.png"))
        entry_bg_1 = canvas.create_image(
            573.0,
            514.0,
            image=entry_image_1
        )
        entry_1 = Entry(name = "precentageEntry",
            bd=0,
            bg="#D5CDEA",
            highlightthickness=0
        )
        entry_1.place(
            x=546.0,
            y=496.0,
            width=54.0,
            height=34.0
        )

        button_image_5 = PhotoImage(
            file=relative_to_assets("button_5.png"))
        button_5 = Button(
            image=button_image_5,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: present_info_text(window),
            relief="flat"
        )
        button_5.place(
            x=359.0,
            y=496.0,
            width=22.0,
            height=27.0
        )

        canvas.create_text(
            394.0,
            504.0,
            anchor="nw",
            text="Threshold value: ",
            fill="#000000",
            font=("JejuMyeongjo", 16 * -1)
        )

        canvas.create_text(
            452.0,
            73.0,
            anchor="nw",
            text="Data sets list:",
            fill="#000000",
            font=("JejuMyeongjo", 24 * -1)
        )

        listbox = tkinter.Listbox(name='listBoxOfDataSet', height=14, width=85)
        listbox.place(x=441.0, y=160.0, )
        directories = os.listdir(".\\..\\DB\\")
        labelFolderExists = tkinter.Label(name='labelFolderExists', fg="red", bg='#E2D8EF').place(x=810, y=570)
        labelInfo = tkinter.Label(name='labelInfo', fg="black", bg='#E2D8EF').place(x=300, y=570)

        for directory in directories:
            listbox.insert(0, directory)

        window.resizable(False, False)
        window.mainloop()
