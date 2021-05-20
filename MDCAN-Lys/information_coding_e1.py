import numpy as np

def one_hot(data, windows=16):      # define input string
    data = data
    length = len(data)
    data_X = np.zeros((length, 2*windows+1, 21))
    data_Y = []
    for i in range(length):
        x = data[i].split()
        data_Y.append(int(x[1]))
        alphabet = 'ACDEFGHIKLMNPQRSTVWY-BJOUXZ'
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        integer_encoded = [char_to_int[char] for char in x[2]]
        # one hot encode
        j = 0
        for value in integer_encoded:
            if value in [21, 22, 23, 24, 25, 26]:
                for k in range(21):
                    data_X[i][j][k] = 0.05
            else:
                data_X[i][j][value] = 1.0
            j = j + 1
    data_Y = np.array(data_Y)

    return data_X, data_Y


def Phy_Chem_Inf(data, windows=16):
    letterDict = {}
    letterDict["A"] = [-0.591, -1.302, -0.733, 1.570, -0.146]
    letterDict["C"] = [-1.343, 0.465, -0.862, -1.020, -0.255]
    letterDict["D"] = [1.050, 0.302, -3.656, -0.259, -3.242]
    letterDict["E"] = [1.357, -1.453, 1.477, 0.113, -0.837]
    letterDict["F"] = [-1.006, -0.590, 1.891, -0.397, 0.412]
    letterDict["G"] = [-0.384, 1.652, 1.330, 1.045, 2.064]
    letterDict["H"] = [0.336, -0.417, -1.673, -1.474, -0.078]
    letterDict["I"] = [-1.239, -0.547, 2.131, 0.393, 0.816]
    letterDict["K"] = [1.831, -0.561, 0.533, -0.277, 1.648]
    letterDict["L"] = [-1.019, -0.987, -1.505, 1.266, -0.912]
    letterDict["M"] = [-0.663, -1.524, 2.219, -1.005, 1.212]
    letterDict["N"] = [0.945, 0.828, 1.299, -0.169, 0.933]
    letterDict["P"] = [0.189, 2.081, -1.628, 0.421, -1.392]
    letterDict["Q"] = [0.931, -0.179, -3.005, -0.503, -1.853]
    letterDict["R"] = [1.538, -0.055, 1.502, 0.440, 2.897]
    letterDict["S"] = [-0.228, 1.399, -4.760, 0.670, -2.647]
    letterDict["T"] = [-0.032, 0.326, 2.213, 0.908, 1.313]
    letterDict["V"] = [-1.337, -0.279, -0.544, 1.242, -1.262]
    letterDict["W"] = [-0.595, 0.009, 0.672, -2.128, -0.184]
    letterDict["Y"] = [0.260, 0.830, 3.097, -0.838, 1.512]
    letterDict["-"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["B"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["J"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["O"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["U"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["X"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    letterDict["Z"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    """
    letterDict = {"A": [-0.59, -1.30, -0.73, 1.57, -0.15],
                  "C": [-1.34, 0.47, -0.86, -1.02, -0.26],
                  "D": [1.05, 0.30, -3.66, -0.26, -3.24],
                  "E": [1.36, -1.45, 1.48, 0.11, -0.84],
                  "F": [-1.01, -0.59, 1.89, -0.40, 0.41],
                  "G": [-0.38, 1.65, 1.33, 1.05, 2.06],
                  "H": [0.34, -0.42, -1.67, -1.47, -0.08],
                  "I": [-1.24, -0.55, 2.13, 0.39, 0.82],
                  "K": [1.83, -0.56, 0.53, -0.28, 1.65],
                  "L": [-1.02, -0.99, -1.51, 1.27, -0.91],
                  "M": [-0.66, -1.52, 2.22, -1.01, 1.21],
                  "N": [0.95, 0.83, 1.30, -0.17, 0.93],
                  "P": [0.19, 2.08, -1.63, 0.42, -1.39],
                  "Q": [0.93, -0.18, -3.01, -0.50, -1.85],
                  "R": [1.54, -0.06, 1.50, 0.44, 2.90],
                  "S": [-0.23, 1.40, -4.76, 0.67, -2.65],
                  "T": [-0.03, 0.33, 2.21, 0.91, 1.31],
                  "V": [-1.34, -0.28, -0.54, 1.24, -1.26],
                  "W": [-0.60, 0.01, 0.67, -2.13, -0.18],
                  "Y": [0.26, 0.83, 3.10, -0.84, 1.51],
                  "-": [0.0, 0.0, 0.0, 0.0, 0.0],
                  "B": [1, 0.565, -1.18, -0.215, -1.155],
                  "J": [-1.13, -0.77, 0.31, 0.83, -0.045],
                  "O": [-0.13, -0.12, 0.6, -0.03, -0.115],
                  "U": [-0.13, -0.12, 0.6, -0.03, -0.115],
                  "X": [-0.13, -0.12, 0.6, -0.03, -0.115],
                  "Z": [1.145, -0.815, -0.765, -0.195, -1.345]
                  }
    """

    # define input string
    data = data
    length = len(data)
    data_X = np.zeros((length, 2*windows+1, 5))
    for i in range(length):
        x = data[i].split()
        j = 0
        for AA in x[2]:
            for index, value in enumerate(letterDict[AA]):
                data_X[i][j][index] = value
            j = j + 1

    return data_X


def Structure_Inf(data, windows=16):

    # define input string
    data = data
    length = len(data)
    data_X = np.zeros((length, 2*windows+1, 8))
    for i in range(length):
        x = data[i].split()

        f_r = open("./dataset/Structure_information/%s.i1" % x[0], "r", encoding='utf-8')
        lines = f_r.readlines()
        List = []
        for line in lines:
            z = line.split()
            if z[0] != '#':
                List.append(line)
        f_r.close()

        k = List[int(x[3])].split()
        if int(k[0]) != int(x[3]) + 1:
            exit()
        j = 0
        offset = 0
        for AA in x[2]:
            if AA != '-':
                value = List[int(x[3]) - windows + offset].split()
                data_X[i][j][0] = value[4]
                data_X[i][j][1] = value[5]
                data_X[i][j][2] = value[6]
                data_X[i][j][3] = value[7]
                data_X[i][j][4] = value[8]
                data_X[i][j][5] = value[14]
                data_X[i][j][6] = value[13]
                data_X[i][j][7] = value[12]
            else:
                data_X[i][j][0] = 0.0
                data_X[i][j][1] = 0.0
                data_X[i][j][2] = 0.0
                data_X[i][j][3] = 0.0
                data_X[i][j][4] = 0.0
                data_X[i][j][5] = 0.0
                data_X[i][j][6] = 0.0
                data_X[i][j][7] = 0.0
            j = j + 1
            offset = offset + 1

    return data_X

