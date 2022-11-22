import networkx as nx
import numpy as np

# A1= [[0,0.5,0.775], [1,2,0], [1,0,1.7]]
# A = np.array([np.array(x) for x in A1])
# print(A)
# type(A)
# G =  nx.from_numpy_matrix(A, create_using=nx.MultiGraph)
val1 = 0
val2 = 1
val3 = 3

# map = {"name1" : {"freqBAlpha" : {"mat:" : val1, "mat2" : val2}, "freqBeta" : {"mat:" : val1, "mat2" : val2}}}
# map1 = {"name2" : {"freqBAlpha" : {"mat:" : val1, "mat2" : val2}, "freqBeta" : {"mat:" : val1, "mat2" : val2}}}
# map.update(map1)
# for name in map:
#     for fb in map[name]:
#         print(fb)
#         print(map[name][fb])


conclusion_matrix = {  # create template for json file
    "patient_name": "",
    "fb_matrix":
        {
            "frequancy_band_name": "",
            "ssr_based_F_testMat": []
        }
}

for i in range(3):
    conclusion_matrix['patient_name'] = str(i)
    for j in range(2):
        conclusion_matrix['fb_matrix']['frequancy_band_name'] = "a"
        conclusion_matrix['fb_matrix']['ssr_based_F_testMat'] = []
print(conclusion_matrix)

# nx.draw(G)