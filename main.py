# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import os
from utilities.json_creator import OutputHandler as jh



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def AvarageMatrix(matrixList):
    # a = {'a':[[1,1,1],[1,1,1],[1,1,1]],'b':[[2,2,2],[2,2,2],[2,2,2]]}
    res = np.zeros((19,19))

    #add all the matrix in the collection
    for x in matrixList:
       res = [[res[i][j] + x[i][j] for j in range(len(res[0]))] for i in range(len(res))]

    t = np.multiply(res, 1 / len(matrixList))

    # result = [[a['a'][i][j] + a['b'][i][j]  for j in range(len(a['a'][0]))] for i in range(len(a['a']))]
    # t = 5
    return t


def BuildFromDir(path):
    files = os.listdir(path)
    numberOfPatint = 0
    for file in files:
        if file.endswith("csv"):
            numberOfPatint = numberOfPatint +1

    mapOfData = {}
    ssr_based_F_testList = []
    ssr_chi2testList = []
    lrtestList = []
    params_ftestList = []

    for file in files:
        if file.endswith("csv"):
            df = pd.read_csv(path+file, parse_dates=['0'])
            colNumber = df.shape[1]-1
            ssr_based_F_testMat = np.zeros((colNumber, colNumber))
            ssr_chi2testMat = np.zeros((colNumber, colNumber))
            lrtestMat = np.zeros((colNumber, colNumber))
            params_ftestMat = np.zeros((colNumber, colNumber))

            for i in range(0,colNumber):
                for j in range(0,colNumber):
                    y = grangercausalitytests(df[[str(i), str(j)]], maxlag=2)

                    ####data from the first part
                    dataFromTest = y[1][0]
                    # type(dataFromTest)
                    ssr_ftest = dataFromTest["ssr_ftest"]
                    # t = ssr_ftest[0]
                    ssr_based_F_testMat[i][j] = ssr_ftest[0]
                    ssr_chi2test = dataFromTest["ssr_chi2test"]
                    ssr_chi2testMat[i][j] = ssr_chi2test[0]
                    lrtest = dataFromTest["lrtest"]
                    lrtestMat[i][j] = lrtest[0]
                    params_ftest = dataFromTest["params_ftest"]
                    params_ftestMat[i][j] = params_ftest[0]

            mapOfData[file] = [("ssr_based_F_testMat",ssr_based_F_testMat),("ssr_chi2testMat",ssr_chi2testMat),("lrtestMat",lrtestMat),("params_ftestMat",params_ftestMat)]
            print(file)
          # shani you need to add here fill json for the map above
            # map of patient name -> couples of 4 matrices and their names

            ssr_based_F_testList.append(ssr_based_F_testMat)
            ssr_chi2testList.append(ssr_chi2testMat)
            lrtestList.append(lrtestMat)

            params_ftestList.append(params_ftestMat)
    jsonC = jh()
    print(jsonC.martix_to_json)
    jsonC.martix_to_json(mapOfData)
    return mapOfData ,ssr_based_F_testList ,ssr_chi2testList,lrtestList,params_ftestList

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    mapOfData ,ssr_based_F_testList ,ssr_chi2testList,lrtestList, params_ftestList = BuildFromDir("C:\\Users\\zivke\\OneDrive\\Documents\\eeg_recording\\Control_part2\\")
    a = {'a':[[1,1,1],[1,1,1],[1,1,1]],'b':[[2,2,2],[2,2,2],[2,2,2]]}
    b = {'a':{'b':[1,1,1],'e':[1,1,1]},'c':{'d':[1,1,1]}}
    k = b['a']
    lists = []
    a1 = np.array([[1,1,1],[1,1,1],[1,1,1]])
    a2 = np.array([[2,2,2],[2,2,2],[2,2,2]])
    ssr_chi2testMat = np.zeros((3, 3))
    lists.append(a1)
    lists.append(a2)
    jsonC = jh()
    # average of all patients
    ssr_based_F_testAvgMatrix = AvarageMatrix(ssr_based_F_testList)
    jsonC.martix_to_csv(ssr_based_F_testAvgMatrix, "ssr_based_F_testAvgMatrix")
    ssr_chi2testMatrix = AvarageMatrix(ssr_chi2testList)
    jsonC.martix_to_csv(ssr_based_F_testAvgMatrix, "ssr_chi2testMatrix")
    lrtestMatrix = AvarageMatrix(lrtestList)
    jsonC.martix_to_csv(ssr_based_F_testAvgMatrix, "lrtestMatrix")
    params_ftestMatrix = AvarageMatrix(params_ftestList)
    jsonC.martix_to_csv(ssr_based_F_testAvgMatrix, "params_ftestMatrix")

stop = 1

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
