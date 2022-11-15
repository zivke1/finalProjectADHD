# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os

import mne
import pywt
from scipy import signal
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




def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    s = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # new_items = [x if x % 2 else None for x in s]
    for index, item in enumerate(s):
        for ind, it in enumerate(item):
            if not (it % 2):
                s[index][ind] = None


    Hz = 128
    timeOfWindow = 3
    win = Hz *timeOfWindow
    df = pd.read_csv("C:\\Users\\zivke\\OneDrive\\Documents\\eeg_recording\\Control_part2\\Control_part1_1_Group_Control.csv", parse_dates=['0'])
    # pywt.swt(df["0"],)
    data = np.loadtxt('C:\\Users\\zivke\\OneDrive\\Documents\\eeg_recording\\Control_part2\\data.txt')
    d0 = pd.Series(df['0']).to_numpy()
    d1 = d0.astype(np.float)
    colNumber = df.shape[1] - 1
    # freqs, psd = signal.welch(d1, Hz, nperseg=win)
    dicOfWindows = {}
    for i in range(colNumber):
            l1 = pd.Series(df[str(i)]).to_numpy()
            l1 = l1.astype(np.float)
            dicOfWindows[i] = [l1[j:j + Hz*timeOfWindow] for j in range(0, len(df[str(i)]), Hz*timeOfWindow)]


    # for each patint dict key = name val dict
    patint ={}
    patint["first"] = {}
    for i in range(colNumber):
        # for each electrode dict key = number val dict
        patint["first"][i] = {}
        # for each brainwave list of window value
        patint["first"][i]['alphaList'] = []
        patint["first"][i]['betaList'] = []
        patint["first"][i]['gammaList'] = []
        patint["first"][i]['thetaList'] = []
        patint["first"][i]['deltaList'] = []
        listsOfElct = dicOfWindows[i]
        for list in listsOfElct:
            delta = bandpower(list,Hz,[0.5, 4],384)
            theta = bandpower(list, Hz, [4, 8], 384)
            alpha = bandpower(list, Hz, [8, 12], 384)
            beta = bandpower(list, Hz, [12, 30], 384)
            gamma= bandpower(list, Hz, [30, 100], 384)
            patint["first"][i]['alphaList'].append(alpha)
            patint["first"][i]['betaList'].append(beta)
            patint["first"][i]['gammaList'].append(gamma)
            patint["first"][i]['thetaList'].append(theta)
            patint["first"][i]['deltaList'].append(delta)


    Delta = bandpower(dicOfWindows[0][0],Hz,[0.5, 4],384)
    Beta = bandpower(dicOfWindows[0][0], Hz, [12, 30], 384)
    # wp = pywt.WaveletPacket(data=df["0"], wavelet='db4', mode='symmetric', maxlevel=4)


    # mapOfData ,ssr_based_F_testList ,ssr_chi2testList,lrtestList, params_ftestList = BuildFromDir("C:\\Users\\zivke\\OneDrive\\Documents\\eeg_recording\\Control_part2\\")
    # a = {'a':[[1,1,1],[1,1,1],[1,1,1]],'b':[[2,2,2],[2,2,2],[2,2,2]]}
    # b = {'a':{'b':[1,1,1],'e':[1,1,1]},'c':{'d':[1,1,1]}}
    # k = b['a']
    # lists = []
    # a1 = np.array([[1,1,1],[1,1,1],[1,1,1]])
    # a2 = np.array([[2,2,2],[2,2,2],[2,2,2]])
    # ssr_chi2testMat = np.zeros((3, 3))
    # lists.append(a1)
    # lists.append(a2)
    # jsonC = jh()
    # # average of all patients
    # ssr_based_F_testAvgMatrix = AvarageMatrix(ssr_based_F_testList)
    # jsonC.martix_to_csv(ssr_based_F_testAvgMatrix, "ssr_based_F_testAvgMatrix")
    # ssr_chi2testMatrix = AvarageMatrix(ssr_chi2testList)
    # jsonC.martix_to_csv(ssr_based_F_testAvgMatrix, "ssr_chi2testMatrix")
    # lrtestMatrix = AvarageMatrix(lrtestList)
    # jsonC.martix_to_csv(ssr_based_F_testAvgMatrix, "lrtestMatrix")
    # params_ftestMatrix = AvarageMatrix(params_ftestList)
    # jsonC.martix_to_csv(ssr_based_F_testAvgMatrix, "params_ftestMatrix")

stop = 1

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
