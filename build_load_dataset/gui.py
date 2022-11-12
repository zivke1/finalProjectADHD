# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer
import tkinter
from pathlib import Path
import os
#for train data set
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utilities.json_creator import OutputHandler as jh
from utilities.GeneralFunction import AvarageMatrix
import build_generate_graphs.gui as generate_graphs_win
import build_analyze_data.gui as analyze_data_win


# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from tkinter import filedialog as fd
import sqlite3


from build_load_dataset.LoadDataSetLogic import LoadDataSetLogic

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# added

def upload_data_set(listBox=None):
    filePathName = fd.askdirectory()

    listBox.insert(0, filePathName)



def train_model_press(listBox=None , parent = None):
    filePathName = listBox.selection_get()
    filePathNameADHD = filePathName + "/ADHD/"
    filePathNameControl = filePathName + "/Control/"

    path = os.getcwd()
    splits = filePathName.split("/")
    folderName = splits[len(splits) - 1]
    # loadDataSet = LoadDataSetLogic()
    path = path.removesuffix("build_main_page")
    path = path + "DB\\" + folderName
    ret = os.path.isdir(path)
    if ret == True:
        print('The folder already exist')
        # parent.labelFolderExists['text'] = "aaa"
        parent.children['labelFolderExists'].config(text = "The folder already exist")
        # parent.__class__.label.place(x=371.0,
                                     # y=499.0, )
        return

    parent.children['labelFolderExists'].config(text="")
    os.mkdir(path)

    mapOfDataADHD, ssr_based_F_testADHDList, ssr_chi2testADHDList, lrtestADHDList, params_ftestADHDList = LoadDataSetLogic.BuildFromDir(
        filePathNameADHD)



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
        filePathNameControl)




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


"""
    print("train_model_press")
    #take each point in the matrix
    Iris = datasets.load_iris()#the data set i need to take the
    irisData = Iris.data[50:150]
    trueLabels = Iris.target[50:150]  # true labeling

    X_train, X_test, trainLabels, testLabels = train_test_split(irisData, trueLabels, test_size=0.40)#take the testing presangeges from the combobox
    plt.plot()
    plt.title("The Iris Dataset labels")
    plt.xlabel(Iris.feature_names[0])
    plt.ylabel(Iris.feature_names[1])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=trainLabels, s=30, label="train")

    # generate a linear SVM model and fit the train data
    svClassifier = SVC(kernel='rbf', C=1000)
    svClassifier.fit(X_train, trainLabels)
    # get and plot the support vectors
    suppportVectors = svClassifier.support_vectors_
    plt.scatter(suppportVectors[:, 0], suppportVectors[:, 1], c="red", marker="+", s=200, label="support vectors")
    # predict the labels of the test vectors
    predictedLabels = svClassifier.predict(X_test)
    #i cant print the data and see something because it has more then 2D
    plt.scatter(X_test[:, 0], X_test[:, 1], c=predictedLabels, marker="x", s=50, label="test")
    plt.legend()
    plt.show()


######confusin matrix build
    table = np.zeros((2, 2), dtype=np.int32)

    for i in range(40):
        if ((predictedLabels[i] == testLabels[i]) and testLabels[i] == 1):  # add one to TP
            table[0, 0] += 1
        elif (predictedLabels[i] == testLabels[i] and testLabels[i] == 2):  # add to TN
            table[1, 1] += 1
        elif (predictedLabels[i] != testLabels[i] and testLabels[i] == 1):  # add to FN
            table[0, 1] += 1
        elif (predictedLabels[i] != testLabels[i] and testLabels[i] == 2):  # add to FP
            table[1, 0] += 1

    print("TP ")
    print(table[0][0])
    print("FN  ")
    print(table[0][1])
    print("FP ")
    print(table[1][0])
    print("TN ")
    print(table[1][1])
"""
class win:
    def __init__(self, *args, **kwargs):
        window = Tk()
        window.geometry("1170x687")
        window.configure(bg="#FFFFFF")

        canvas = Canvas(
            window,
            bg="#FFFFFF",
            height=700,
            width=1186,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        canvas.place(x=0, y=0)
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


    # upload data set button
        button_image_1 = PhotoImage(
            file=relative_to_assets("button_1.png"))
        button_1 = Button(
            image=button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda:upload_data_set(listBox=listbox),
            relief="flat"
        )
        button_1.place(
            x=371.0,
            y=499.0,
            width=234.0,
            height=48.0
        )

        button_image_2 = PhotoImage(
            file=relative_to_assets("button_2.png"))
        button_2 = Button(
            image=button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command= lambda:train_model_press(listBox=listbox,parent = window),
            relief="flat"
        )
        button_2.place(
            x=797.0,
            y=499.0,
            width=234.0,
            height=48.0
        )

        button_image_3 = PhotoImage(
            file=relative_to_assets("button_3.png"))
        button_3 = Button(
            image=button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: {
                # print("button_3 clicked")
                    window.destroy():
                    generate_graphs_win.win()
                             },
            relief="flat"
        )
        button_3.place(
            x=34.0,
            y=154.0,
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
            command=lambda: {
                    window.destroy():
                    analyze_data_win.win()
            },
            relief="flat"
        )
        button_5.place(
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

        button_image_6 = PhotoImage(
            file=relative_to_assets("button_6.png"))
        button_6 = Button(
            image=button_image_6,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: print("button_6 clicked"),
            relief="flat"
        )
        button_6.place(
            x=432.0,
            y=117.0,
            width=499.0,
            height=312.0
        )

        canvas.create_rectangle(
            483.0,
            146.0,
            681.0,
            280.0,
            fill="#FFFFFF",
            outline="")

        listbox = tkinter.Listbox(height=15, width=70)
        listbox.place(x=470.0, y=146.0, )

        labelFolderExists = tkinter.Label(name='labelFolderExists',  fg="red", bg='#E2D8EF').place(x=810,
                                                                                       y=570)
        window.resizable(False, False)
        window.mainloop()




if __name__ == "__main__":
    # Create button, it will change label text
    # button = Button(window, text="click Me", command=show).pack()

    # Create a database or connect to one
    conn = sqlite3.connect('data_sets.db')
    # Create cursor
    cursor = conn.cursor()

    # # Create table - only for the first time
    # cursor.execute("""CREATE TABLE dataSets(
    #         dataSetCsvFile blob
    #     )""")



    # Commit changes
    conn.commit()
    #Close connection
    conn.close()
    win()
