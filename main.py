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



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def AvarageMatrix(matrixList):
    a = {'a':[[1,1,1],[1,1,1],[1,1,1]],'b':[[2,2,2],[2,2,2],[2,2,2]]}
    res = np.zeros((3,3))

    #add all the matrix in the collection
    for x in matrixList:
       res = [[res[i][j] + x[i][j] for j in range(len(res[0]))] for i in range(len(res))]

    t = np.multiply(res, 1 / len(matrixList))

    result = [[a['a'][i][j] + a['b'][i][j]  for j in range(len(a['a'][0]))] for i in range(len(a['a']))]
    # t = 5
    return result


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
            ssr_based_F_testList.append(ssr_based_F_testMat)
            ssr_chi2testList.append(ssr_chi2testMat)
            lrtestList.append(lrtestMat)

            params_ftestList.append(params_ftestMat)


    return mapOfData ,ssr_based_F_testList ,ssr_chi2testList,lrtestList,params_ftestList

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    mapOfData ,ssr_based_F_testList ,ssr_chi2testList,lrtestList, params_ftestList = BuildFromDir(r"C:\Users\zivke\OneDrive\Documents\eeg_recording\TwoPersonCheck\\")
    a = {'a':[[1,1,1],[1,1,1],[1,1,1]],'b':[[2,2,2],[2,2,2],[2,2,2]]}
    b = {'a':{'b':[1,1,1],'e':[1,1,1]},'c':{'d':[1,1,1]}}
    k = b['a']
    lists = []
    a1 = np.array([[1,1,1],[1,1,1],[1,1,1]])
    a2 = np.array([[2,2,2],[2,2,2],[2,2,2]])
    ssr_chi2testMat = np.zeros((3, 3))
    lists.append(a1)
    lists.append(a2)
    ssr_based_F_testAvgMatrix = AvarageMatrix(ssr_based_F_testList)
    ssr_chi2testMatrix = AvarageMatrix(ssr_chi2testList)
    lrtestMatrix = AvarageMatrix(lrtestList)
    params_ftestMatrix = AvarageMatrix(params_ftestList)



'''
    df = pd.read_csv("C:\\Users\\zivke\\OneDrive\\Documents\\eeg_recording\\ADHD_part1\\ADHD_part1_10_Group_Control.csv", parse_dates=['0'])
    rowNumber = len(df.index)
    colNumber = df.shape[1]-1
    y = grangercausalitytests(df[['0', '0']], maxlag=2)

    dataFromTest = y[1][0]
    type(dataFromTest)
    asss = dataFromTest["ssr_ftest"]
    # type(asss)
    t = asss[0]
    ssr_based_F_testMat = np.zeros((colNumber,colNumber))
    ssr_chi2testMat = np.zeros((colNumber,colNumber))
    lrtestMat = np.zeros((colNumber,colNumber))
    params_ftestMat = np.zeros((colNumber,colNumber))

    for i in range (colNumber):
        for j in range (colNumber):
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



    print(ssr_based_F_testMat)
    print(ssr_chi2testMat)
    print(lrtestMat)
    print(params_ftestMat)

    print(ssr_based_F_testMat[0][0])
'''

stop = 1

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
