
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer

import tkinter
from pathlib import Path
from tkinter import ttk, LEFT
from threading import Thread
import shutil

from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage,Label
import build_generate_graphs.gui as generate_graphs_win
from tkinter import filedialog as fd
import os
from utilities.json_creator import OutputHandler as jh
from utilities.GeneralFunction import AvarageMatrix
from PIL import ImageTk, Image
from build_load_dataset.LoadDataSetLogic import LoadDataSetLogic

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")
import functools
controlGroupPath = ""
treatmentGroupPath = ""

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def upload_data_set(win = None ,id = 0):

    global controlGroupPath
    global treatmentGroupPath
    filePathName = fd.askdirectory()
    if id == 1:
        controlGroupPath = filePathName
        splitPath = filePathName.split("/")
        win.children['checkMarkG1'].config(text = splitPath[splitPath.__len__()-1])
    elif id == 2:
        treatmentGroupPath = filePathName
        splitPath = filePathName.split("/")
        win.children['checkMarkG2'].config(text=splitPath[splitPath.__len__() - 1])

    if controlGroupPath == treatmentGroupPath:
        win.children['warnningSameFolder'].config(text="You choose the same folder \nfor treatment and control")
    else:
        win.children['warnningSameFolder'].config(text="")


def openThread(parent=None):
    thread = Thread(target=generate_graphs_press, args=(parent,))
    thread.start()

def generate_graphs_press(parent = None):
    parent.children['labelFinish'].config(text="")
    parent.children["progBar"]['value'] = 0
    entryHz = parent.children["entryHz"].get()
    entryWinSec = parent.children["entryWinSec"].get()
    entryG1 = parent.children["entryG1"].get()
    entryG2 = parent.children["entryG2"].get()

    if controlGroupPath == '' or treatmentGroupPath == '' or entryHz == '' or entryWinSec == '' or entryG1 == '' or entryG2== '':
        parent.children['labelFolderExists'].config(text="You must enter all the fields")
        return
    parent.children['labelFolderExists'].config(text="")
    filePathNameTreatment  = treatmentGroupPath + "//"
    filePathNameControl = controlGroupPath + "//"

    path = os.getcwd()
    # splits = filePathName.split("/")
    # folderName = splits[len(splits) - 1]
    folderName = entryG1+"_"+entryG2+"_Hz"+entryHz+"_WinSec"+entryWinSec
    # loadDataSet = LoadDataSetLogic()
    path = path.removesuffix("build_main_page")
    path = path + "DB\\" + folderName
    ret = os.path.isdir(path)
    if ret == True:
        print('The folder already exist')
        shutil.rmtree(path)  # remove non-empty folder
        # parent.labelFolderExists['text'] = "aaa"
        parent.children['labelFolderExists'].config(text = "The folder already exist")
        # parent.__class__.label.place(x=371.0,
                                     # y=499.0, )


    parent.children['labelFolderExists'].config(text="")
    os.mkdir(path)

    mapOfDataADHD, ssr_based_F_testADHDList, ssr_chi2testADHDList, lrtestADHDList, params_ftestADHDList = LoadDataSetLogic.BuildFromDir(
        filePathNameTreatment, entryWinSec , entryHz ,parent.children['progBar'])



    jsonC = jh()
    print(jsonC.martix_to_json)
    jsonC.martix_to_json(mapOfDataADHD, "conclusionMatrixADHD",folderName)

    # average of all adhd patients
    ssr_based_F_testAvgADHDMatrix = AvarageMatrix(ssr_based_F_testADHDList)
    jsonC.martix_to_csv(ssr_based_F_testAvgADHDMatrix, "ssr_based_F_testAvgADHDMatrix",folderName)
    ssr_chi2testAvgADHDMatrix = AvarageMatrix(ssr_chi2testADHDList)
    jsonC.martix_to_csv(ssr_chi2testAvgADHDMatrix, "ssr_chi2testAvgADHDMatrix",folderName)
    lrtestAvgADHDMatrix = AvarageMatrix(lrtestADHDList)
    jsonC.martix_to_csv(lrtestAvgADHDMatrix, "lrtestAvgADHDMatrix",folderName)
    params_ftestAvgADHDMatrix = AvarageMatrix(params_ftestADHDList)
    jsonC.martix_to_csv(params_ftestAvgADHDMatrix, "params_ftestAvgADHDMatrix",folderName)

    mapOfDataControl, ssr_based_F_testControlList, ssr_chi2testControlList, lrtestControlList, params_ftestControlList = LoadDataSetLogic.BuildFromDir(
        filePathNameControl, entryWinSec , entryHz ,parent.children['progBar'])




    jsonC.martix_to_json(mapOfDataControl, "conclusionMatrixControl",folderName)

    # average of all patients control
    ssr_based_F_testAvgControlMatrix = AvarageMatrix(ssr_based_F_testControlList)
    jsonC.martix_to_csv(ssr_based_F_testAvgControlMatrix, "ssr_based_F_testAvgControlMatrix",folderName)
    ssr_chi2testAvgControlMatrix = AvarageMatrix(ssr_chi2testControlList)
    jsonC.martix_to_csv(ssr_chi2testAvgControlMatrix, "ssr_chi2testAvgControlMatrix",folderName)
    lrtestAvgControlMatrix = AvarageMatrix(lrtestControlList)
    jsonC.martix_to_csv(lrtestAvgControlMatrix, "lrtestAvgControlMatrix",folderName)
    params_ftestAvgControlMatrix = AvarageMatrix(params_ftestControlList)
    jsonC.martix_to_csv(params_ftestAvgControlMatrix, "params_ftestAvgControlMatrix",folderName)
    parent.children['progBar']['value']= 100
    parent.children['labelFinish'].config(text = "Finish upload data set\nThe folder "+folderName +"\nwas created\ngo to generate graphs and analyse this data set")
    return


def on_closing(root):
    dir = '..\DB2\graphs'
    # check first if dir exist
    # delete dir content
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    root.destroy()

class win:
    def __init__(self, *args, **kwargs):
        window = Tk()
        window.title("Load Data Set")
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

        entry_image_1 = PhotoImage(
            file=relative_to_assets("entry_1.png"))
        entry_bg_1 = canvas.create_image(
            643.0,
            379.0,
            image=entry_image_1
        )
        entry_1 = Entry(name = "entryHz",
            bd=0,
            bg="#D5CDEA",
            highlightthickness=0
        )
        entry_1.place(
            x=616.0,
            y=361.0,
            width=54.0,
            height=34.0
        )

        entry_image_2 = PhotoImage(
            file=relative_to_assets("entry_2.png"))
        entry_bg_2 = canvas.create_image(
            643.0,
            303.0,
            image=entry_image_2
        )
        entry_2 = Entry(name = "entryWinSec",
            bd=0,
            bg="#D5CDEA",
            highlightthickness=0
        )
        entry_2.place(
            x=616.0,
            y=285.0,
            width=54.0,
            height=34.0
        )

        canvas.create_text(
            334.0,
            294.0,
            anchor="nw",
            text="Choose sliding window (seconds):",
            fill="#000000",
            font=("JejuMyeongjo", 16 * -1)
        )

        canvas.create_text(
            570.0,
            294.0,
            anchor="nw",
            text="*",
            fill="red",
            font=("JejuMyeongjo", 16 * -1,)

        )

        entry_image_3 = PhotoImage(
            file=relative_to_assets("entry_3.png"))
        entry_bg_3 = canvas.create_image(
            674.0,
            114.0,
            image=entry_image_3
        )
        entry_3 = Entry(name = "entryG1",
            bd=0,
            bg="#D5CDEA",
            highlightthickness=0
        )
        entry_3.place(
            x=595.0,
            y=96.0,
            width=158.0,
            height=34.0
        )

        canvas.create_text(
            334.0,
            105.0,
            anchor="nw",
            text="Enter name for control group:",
            fill="#000000",
            font=("JejuMyeongjo", 16 * -1)
        )
        canvas.create_text(
            555.0,
            30.0,
            anchor="nw",
            text="Load raw EEG data set",
            fill="#000000",
            font=("JejuMyeongjo", 24 * -1)
        )


        canvas.create_text(
            555.0,
            189.0,
            anchor="nw",
            text="*",
            fill="red",
            font=("JejuMyeongjo", 16 * -1,)

        )

        canvas.create_text(
            535.0,
            105.0,
            anchor="nw",
            text="*",
            fill="red",
            font=("JejuMyeongjo", 16 * -1,)

        )
        canvas.create_text(
            334.0,
            369.0,
            anchor="nw",
            text="Sampling frequency (Hz):",
            fill="#000000",
            font=("JejuMyeongjo", 16 * -1)
        )
        canvas.create_text(
            508.0,
            369.0,
            anchor="nw",
            text="*",
            fill="red",
            font=("JejuMyeongjo", 16 * -1,)

        )

        button_image_1 = PhotoImage(
            file=relative_to_assets("button_1.png"))
        button_1 = Button(
            image=button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: upload_data_set(win=window,id =1),
            relief="flat"
        )
        button_1.place(
            x=841.0,
            y=92.0,
            width=234.0,
            height=48.0
        )

        entry_image_4 = PhotoImage(
            file=relative_to_assets("entry_4.png"))
        entry_bg_4 = canvas.create_image(
            674.0,
            198.0,
            image=entry_image_4
        )
        entry_4 = Entry(name = "entryG2",
            bd=0,
            bg="#D5CDEA",
            highlightthickness=0
        )
        entry_4.place(
            x=595.0,
            y=180.0,
            width=158.0,
            height=34.0
        )

        canvas.create_text(
            334.0,
            189.0,
            anchor="nw",
            text="Enter name for treatment group:",
            fill="#000000",
            font=("JejuMyeongjo", 16 * -1)
        )

        button_image_2 = PhotoImage(
            file=relative_to_assets("button_2.png"))
        button_2 = Button(
            image=button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: upload_data_set(win=window,id =2),
            relief="flat"
        )
        button_2.place(
            x=841.0,
            y=176.0,
            width=234.0,
            height=48.0
        )

        button_image_3 = PhotoImage(
            file=relative_to_assets("button_3.png"))
        button_3 = Button(
            image=button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command= lambda:openThread(parent=window),
            relief="flat"
        )
        button_3.place(
            x=789.0,
            y=490.0,
            width=234.0,
            height=48.0
        )

        button_image_4 = PhotoImage(
            file=relative_to_assets("button_4.png"))
        button_4 = Button(
            image=button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: {
                window.destroy():
                generate_graphs_win.win()
            },
            relief="flat"
        )
        button_4.place(
            x=34.0,
            y=92.0,
            width=168.0,
            height=37.0
        )

        button_image_5 = PhotoImage(
            file=relative_to_assets("button_5.png"))
        button_5 = Button(
            image=button_image_5,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: print("button_5 clicked"),
            relief="flat"
        )
        button_5.place(
            x=34.0,
            y=154.0,
            width=168.0,
            height=37.0
        )

        labelG1 = tkinter.Label(name='checkMarkG1', fg="black", bg='#E2D8EF',
                                font=("JejuMyeongjo", 16 * -1))
        labelG1.place(x=1080, y=100)

        labelG2 = tkinter.Label(name='checkMarkG2', fg="black", bg='#E2D8EF',
                                          font=("JejuMyeongjo", 16 * -1))
        labelG2.place(x=1080, y=185)

        warnningSameFolder = tkinter.Label(name='warnningSameFolder', fg="red", bg='#E2D8EF',justify=LEFT,
                                font=("JejuMyeongjo", 16 * -1))
        warnningSameFolder.place(x=850, y=250)

        pb = ttk.Progressbar(
            window,name = 'progBar',
            orient='horizontal',
            mode='determinate',
            length=280
        )
        pb.place(x=530.0, y=440.0, )
        labelFolderExists = tkinter.Label(name='labelFolderExists', fg="red", bg='#E2D8EF').place(x=810, y=570)
        labelFolderExists = tkinter.Label(name='labelFinish', fg="black",justify=LEFT, bg='#E2D8EF',font=("JejuMyeongjo", 16 * -1))
        labelFolderExists.place(x=650, y=560)

        window.resizable(False, False)
        on_close_with_params = functools.partial(on_closing, window)
        window.protocol("WM_DELETE_WINDOW", on_close_with_params)
        window.mainloop()
