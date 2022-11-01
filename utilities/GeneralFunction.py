import numpy as np

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