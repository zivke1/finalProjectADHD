import json
import csv
import os
import pandas as pd
import numpy as np
# write route data topic to json file from a telemetry file- up to 100 spots

class OutputHandler:

    def __init__(self):
        self.telem_path = "/home/maya/Rocky_Simulations/map/metria.csv"   # currently not in use
        # self.flag_first_row = 1

    def martix_to_json(self, mapOfData, outputFileName):
        conclusion_matrix = {               # create template for json file
                "patient_name": "",
                "ssr_based_F_testMat": [],
                "ssr_chi2testMat": [],
                "lrtestMat": [],
                "params_ftestMat": []
            }

        jpath = outputFileName +'.json'
        j_file = open(jpath, 'w')
        j_file.write('{\n"Patients": [\n')

        for patient in mapOfData:           # insert data into struct
            conclusion_matrix['patient_name'] = patient.strip(".csv")
            conclusion_matrix['ssr_based_F_testMat'] = mapOfData[patient][0][1].tolist()  # get the second element in the tuple which is the matrix
            conclusion_matrix['ssr_chi2testMat'] = mapOfData[patient][1][1].tolist()
            conclusion_matrix['lrtestMat'] = mapOfData[patient][2][1].tolist()
            conclusion_matrix['params_ftestMat'] = mapOfData[patient][3][1].tolist()
            j_file.write('\t')
            json.dump(conclusion_matrix, j_file)
            j_file.write(',\n\n')

        j_file.seek(j_file.tell() - 5, os.SEEK_SET)  # go back 3 from last position
        j_file.write('\n\t]\n}')
        j_file.close()

    def martix_to_csv(self, matrix, name):
        #for patient in mapOfData:
            # convert array into dataframe
        DF = pd.DataFrame(matrix)

        # save the dataframe as a csv file
        DF.to_csv("matrix_"+str(name)+".csv")

            ##### for now save only the first matrix for each patient !!!
