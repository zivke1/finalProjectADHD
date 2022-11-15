
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer
import os
import tkinter
from pathlib import Path
import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
import finalProjectADHD.build_load_dataset.gui as load_dataset_win
import finalProjectADHD.build_analyze_data.gui as analyze_data_win
from finalProjectADHD.utilities.json_creator import OutputHandler as jh

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


def remove_values_from_matrix_under_precentages(data , precentage):
    # this code is remove all the values under the threshold
    for patient in data:
        # p = patient
        ssr_based_F_testMat = patient['ssr_based_F_testMat']
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

    listOf_graphs_ADHD_group = []
    create_graphs(dataADHD, 'ssr_based_F_testMat', listOf_graphs_ADHD_group)  # create graphs for ADHD patients and insert to the list
    jsonH.listOf_graphs_to_json(listOf_graphs_ADHD_group, "ADHD_group_graphs"+precentage, "DB2\graphs")
    #### above ADHD ; below control  ####

    folderPath = ".\\..\\DB\\"+diractory+"\\conclusionMatrixControl.json"
    f = open(folderPath)
    conclusionMatrixControl = json.load(f)
    dataControl = conclusionMatrixControl['Patients']
    remove_values_from_matrix_under_precentages(dataControl ,precentage)

    listOf_graphs_control_group = []
    create_graphs(dataControl, 'ssr_based_F_testMat', listOf_graphs_control_group)  # create graphs for control patients and insert to the list
    jsonH.listOf_graphs_to_json(listOf_graphs_control_group, "control_group_graphs"+precentage, "DB2\graphs")

 # destroy window at the end of the function
    name_of_ADHD_graph_file = "ADHD_group_graphs"
    name_of_control_graph_file = "control_group_graphs"
    parent.destroy()
    analyze_data_win.win(name_of_ADHD_graph_file, name_of_control_graph_file)

def create_graphs(patients_data, matrix_name,listOf_graphs):
    ## create graph from ssr_based_F_testMat
    for patient in patients_data:
        ssr_based_F_testMat = patient[matrix_name]
        # for i in range(len(ssr_based_F_testMat)):
        #     for j in range(len(ssr_based_F_testMat[i])):
        #         if ssr_based_F_testMat[i][j] is None:
        #             ssr_based_F_testMat[i][j] = 0
        listOflist_to_npArray = np.array([np.array(i) for i in ssr_based_F_testMat])

        G = nx.from_numpy_matrix(listOflist_to_npArray, create_using=nx.DiGraph)
        listOf_graphs.append(G)

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

        canvas.create_rectangle(
            234.0,
            0.0,
            1167.0,
            684.0,
            fill="#E2D8EF",
            outline="")

        entry_image_1 = PhotoImage(
            file=relative_to_assets("entry_1.png"))
        entry_bg_1 = canvas.create_image(
            549.0,
            519.0,
            image=entry_image_1
        )
        entry_1 = Entry(name = "precentageEntry",
            bd=0,
            bg="#D5CDEA",
            highlightthickness=0
        )
        entry_1.place(
            x=522.0,
            y=501.0,
            width=54.0,
            height=34.0
        )

        button_image_1 = PhotoImage(
            file=relative_to_assets("button_1.png"))
        button_1 = Button(
            image=button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda:{
                gen_graphs_pressed(parent = window)
            },
            relief="flat"
        )
        button_1.place(
            x=784.0,
            y=493.0,
            width=206.0,
            height=50.0
        )

        button_image_2 = PhotoImage(
            file=relative_to_assets("button_2.png"))
        button_2 = Button(
            image=button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: present_info_text(window),
            relief="flat"
        )
        button_2.place(
            x=335.0,
            y=501.0,
            width=22.0,
            height=27.0
        )

        button_image_3 = PhotoImage(
            file=relative_to_assets("button_3.png"))
        button_3 = Button(
            image=button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: print("button_3 clicked"),
            relief="flat"
        )
        button_3.place(
            x=432.0,
            y=119.0,
            width=499.0,
            height=310.0
        )

        button_image_4 = PhotoImage(
            file=relative_to_assets("button_4.png"))
        button_4 = Button(
            image=button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda:{},
            relief="flat"
        )
        button_4.place(
            x=34.0,
            y=154.0,
            width=168.0,
            height=37.0
        )

        button_image_5 = PhotoImage(
            file=relative_to_assets("button_5.png"))
        button_5 = Button(
            image=button_image_5,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: {
                    window.destroy():
                    load_dataset_win.win()
            },
            relief="flat"
        )
        button_5.place(
            x=34.0,
            y=92.0,
            width=168.0,
            height=37.0
        )

        button_image_6 = PhotoImage(
            file=relative_to_assets("button_6.png"))
        button_6 = Button(
            image=button_image_6,
            borderwidth=0,
            highlightthickness=0,
            command=lambda:{
                    window.destroy():
                    analyze_data_win.win()
            },
            relief="flat"
        )
        button_6.place(
            x=34.0,
            y=215.0,
            width=168.0,
            height=37.0
        )

        canvas.create_text(
            449.0,
            74.0,
            anchor="nw",
            text="Data sets list:",
            fill="#000000",
            font=("JejuMyeongjo", 24 * -1)
        )

        canvas.create_text(
            27.0,
            33.0,
            anchor="nw",
            text="EEG Recordings Analyzer",
            fill="#FFFAFA",
            font=("JejuMyeongjo", 14 * -1)
        )

        canvas.create_text(
            370.0,
            509.0,
            anchor="nw",
            text="Threshold value: ",
            fill="#000000",
            font=("JejuMyeongjo", 16 * -1)
        )

        listbox = tkinter.Listbox(name='listBoxOfDataSet', height=15, width=70)
        listbox.place(x=470.0, y=146.0, )
        directories = os.listdir(".\\..\\DB\\")
        labelFolderExists = tkinter.Label(name='labelFolderExists', fg="red", bg='#E2D8EF').place(x=810,y=570)
        labelInfo = tkinter.Label(name='labelInfo', fg="black", bg='#E2D8EF').place(x=300,y=570)

        for directory in directories:
            listbox.insert(0,directory)

        window.resizable(False, False)
        window.mainloop()

# if __name__ == "__main__":
    # win()