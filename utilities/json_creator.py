import json
import csv
import os
# write route data topic to json file from a telemetry file- up to 100 spots

class JsonHandler:

    def __init__(self):
        self.telem_path = "/home/maya/Rocky_Simulations/map/metria.csv"   # currently not in use
        # self.flag_first_row = 1

    def martix_to_json(self, mapOfData):
        conclusion_matrix = {               # create template for json file
                "patient_name": "",
                "mat1": [],
                "mat2": [],
                "mat3": [],
                "mat4": []
            }

        jpath = 'conclusionMatrix.json'
        j_file = open(jpath, 'w')
        j_file.write('{\n"Patients": [\n')

        for patient in mapOfData:           # insert data into struct
            conclusion_matrix['mat1'] = mapOfData[patient][0][1].tolist()  # get the second element in the tuple which is the matrix
            conclusion_matrix['mat2'] = mapOfData[patient][1][1].tolist()
            conclusion_matrix['mat3'] = mapOfData[patient][2][1].tolist()
            conclusion_matrix['mat4'] = mapOfData[patient][3][1].tolist()
            j_file.write('\t')
            json.dump(conclusion_matrix, j_file)
            j_file.write(',\n\n')

        j_file.seek(j_file.tell() - 3, os.SEEK_SET)  # go back 3 from last position
        j_file.write('\n\t]\n}')
        j_file.close()

'''
with open(telem_path) as telem_file:
    reader = csv.reader(telem_file)
    rows = list(reader)
    rows_num = len(rows)
    count = 0
    telem_file.seek(0)
    for row in reader:
        count += 1
        if flag_first_row:
            flag_first_row = 0
            continue
        lat = row.__getitem__(1)  # column 1
        lon = row.__getitem__(2)  # column 2
        route_dict['A_Coordinate3DType']['A_latitude'] = float(lat)
        route_dict['A_Coordinate3DType']['A_longitude'] = float(lon)
        j_file.write('\t')
        json.dump(route_dict, j_file)
        j_file.write(',\n\n')
        n = rows_num/100
        for i in range(int(n)):   # skip lines
            if count == rows_num:
                break
            next(reader)
            count+=1
'''

