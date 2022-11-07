import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
# from utilities.json_creator.py import OutputHandler as jh
from utilities.json_creator import OutputHandler as jh


class LoadDataSetLogic:

    def BuildFromDir(path):
        files = os.listdir(path)
        numberOfPatint = 0
        for file in files:
            if file.endswith("csv"):
                numberOfPatint = numberOfPatint + 1

        mapOfData = {}
        ssr_based_F_testList = []
        ssr_chi2testList = []
        lrtestList = []
        params_ftestList = []

        for file in files:
            if file.endswith("csv"):
                df = pd.read_csv(path + file, parse_dates=['0'])
                colNumber = df.shape[1] - 1
                ssr_based_F_testMat = np.zeros((colNumber, colNumber))
                ssr_chi2testMat = np.zeros((colNumber, colNumber))
                lrtestMat = np.zeros((colNumber, colNumber))
                params_ftestMat = np.zeros((colNumber, colNumber))

                for i in range(0, colNumber):
                    for j in range(0, colNumber):
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

                mapOfData[file] = [("ssr_based_F_testMat", ssr_based_F_testMat), ("ssr_chi2testMat", ssr_chi2testMat),
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
