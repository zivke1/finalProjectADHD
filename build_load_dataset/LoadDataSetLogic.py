import os
import numpy as np
import pandas as pd
import copy
from statsmodels.tsa.stattools import grangercausalitytests
# from utilities.json_creator.py import OutputHandler as jh
from utilities.json_creator import OutputHandler as jh


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

class LoadDataSetLogic:



    def BuildFromDir(path, winLength , freqHz):
        files = os.listdir(path)
        # numberOfPatint = 0
        # for file in files:
        #     if file.endswith("csv"):
        #         numberOfPatint = numberOfPatint + 1

        mapOfData = {}
        ssr_based_F_testList = []
        ssr_chi2testList = []
        lrtestList = []
        params_ftestList = []
        listOfFrequencyTypes = ['alphaList' ,'betaList','gammaList' ,'thetaList','deltaList']


        for file in files:
            if file.endswith("csv"):
                df = pd.read_csv(path + file, parse_dates=['0'])
                colNumber = df.shape[1] - 1

                ssr_chi2testMat = np.zeros((colNumber, colNumber))
                lrtestMat = np.zeros((colNumber, colNumber))
                params_ftestMat = np.zeros((colNumber, colNumber))

                mapOfElctrodeAndPart = {}
                electrodeToFeq = {}
                for i in range(colNumber):
                    l1 = pd.Series(df[str(i)]).to_numpy()
                    l1 = l1.astype(np.float)
                    listOfParts = [l1[j:j + int(freqHz) * int(winLength)] for j in
                                       range(0, len(df[str(i)]), int(freqHz) * int(winLength))]


                    electrodeToFeq[i] = {}
                    electrodeToFeq[i]['alphaList'] = []
                    electrodeToFeq[i]['betaList'] = []
                    electrodeToFeq[i]['gammaList'] = []
                    electrodeToFeq[i]['thetaList'] = []
                    electrodeToFeq[i]['deltaList'] = []
                    for k in range (listOfParts.__len__()):
                         if listOfParts[k].size != int(freqHz) * int(winLength):
                             continue
                         delta = bandpower(listOfParts[k], int(freqHz), [0.5, 4],  int(winLength))
                         theta = bandpower(listOfParts[k], int(freqHz), [4, 8], int(winLength))
                         alpha = bandpower(listOfParts[k], int(freqHz), [8, 12], int(winLength))
                         beta = bandpower(listOfParts[k], int(freqHz), [12, 30],  int(winLength))
                         gamma = bandpower(listOfParts[k], int(freqHz), [30, 100],  int(winLength))
                         electrodeToFeq[i]['alphaList'].append(alpha)
                         electrodeToFeq[i]['betaList'].append(beta)
                         electrodeToFeq[i]['gammaList'].append(gamma)
                         electrodeToFeq[i]['thetaList'].append(theta)
                         electrodeToFeq[i]['deltaList'].append(delta)
                    #

                mapOfData[file] = {}
                for type in listOfFrequencyTypes:
                    ssr_based_F_testMat = np.zeros((colNumber, colNumber))
                    for i in range(0, colNumber):
                        for j in range(0, colNumber):

                             data = {'0':electrodeToFeq[i][type],'1':electrodeToFeq[j][type]}
                             dataDf = pd.DataFrame(data)
                             # y = grangercausalitytests(df[[str(i), str(j)]], maxlag=2)
                             y = grangercausalitytests(dataDf[['0', '1']], maxlag=2)

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

                #need to add part of type!!
                    mapOfData[file][type] = [("ssr_based_F_testMat", ssr_based_F_testMat), ("ssr_chi2testMat", ssr_chi2testMat),
                                   ("lrtestMat", lrtestMat), ("params_ftestMat", params_ftestMat)]
                print(file)
                # shani you need to add here fill json for the map above
                # map of patient name -> couples of 4 matrices and their names

                ssr_based_F_testList.append(ssr_based_F_testMat)
                ssr_chi2testList.append(ssr_chi2testMat)
                lrtestList.append(lrtestMat)

                params_ftestList.append(params_ftestMat)
        # jsonC = jh()
        # print(jsonC.martix_to_json)
        # jsonC.martix_to_json(mapOfData)
        return mapOfData, ssr_based_F_testList, ssr_chi2testList, lrtestList, params_ftestList
