# Load the data from the .log file
import json
import matplotlib.pyplot as plt
import numpy as np



def plot(dataname, L, observables=True):
    data=json.load(open(dataname))
    # Extract the relevant information

    length = L

    iters=[]
    energy=[]
    sfs_fast = list()
    xAxis_fast = list()

    for iteration in data["Output"]:
        iters.append(iteration["Iteration"])
        energy.append(iteration["Energy"]["Mean"])

    def calcMean(array):
        sum = 0.
        for i in range(15):
            sum += array[-i+0]
        return sum/15.

    def getsf(i):
        sf = list()
        for iteration in data["Output"]:
            sf.append(iteration['Ferro_correlation_function' + str(i)]["Mean"])
        return calcMean(sf)

    plt.plot(iters, energy)
    plt.show()

    if(observables == True):
        for i in range(1, length):
            sfs_fast.append(getsf(i))
            xAxis_fast.append(i)

        plt.plot(xAxis_fast, sfs_fast)
        plt.show()



# Ls should be an array with 4 Elements
def present(Ls, path):

    def plot4me(a, b, text='Energy-Iterations'):
        plt.subplot(221)  # sublot(Anzahl Zeilen Anzahl Spalten Bild Nummer)
        plt.plot(a[0], b[0])
        plt.title(''.join((text, ' ', str(Ls[0]))))
        plt.subplot(222)  # sublot(Anzahl Zeilen Anzahl Spalten Bild Nummer)
        plt.plot(a[1], b[1])
        plt.title(''.join((text, ' ', str(Ls[1]))))
        plt.subplot(223)  # sublot(Anzahl Zeilen Anzahl Spalten Bild Nummer)
        plt.plot(a[2], b[2])
        plt.title(''.join((text, ' ', str(Ls[2]))))
        plt.subplot(224)  # sublot(Anzahl Zeilen Anzahl Spalten Bild Nummer)
        plt.plot(a[3], b[3])
        plt.title(''.join((text, ' ', str(Ls[3]))))
        plt.show()

    iters_list = []
    energy_list = []
    sfs_fast_list = []
    xAxis_fast_list = []
    for i in range(0, len(Ls)):
        l = Ls[i]
        print(''.join((path, '/L', str(l), '.log')))
        data_energy = json.load(open(''.join((path, '/L', str(l), '.log'))))
        data_observ = json.load(open(''.join((path, '/L', str(l), '_estimate.log'))))
        # Extract the relevant information

        length = l

        iters = []
        energy = []
        sfs_fast = list()
        xAxis_fast = list()

        for iteration in data_energy["Output"]:
            iters.append(iteration["Iteration"])
            energy.append(iteration["Energy"]["Mean"])

        def calcMean(array):
            sum = 0.
            for i in range(15):
                sum += array[-i + 0]
            return sum / 15.

        def getsf(i):
            sf = list()
            for iteration in data_observ["Output"]:
                sf.append(iteration['Ferro_correlation_function' + str(i)]["Mean"])
            return calcMean(sf)

        for i in range(1, length):
            sfs_fast.append(getsf(i))
            xAxis_fast.append(i)


        iters_list.append(np.asarray(iters))
        energy_list.append(np.asarray(energy))
        sfs_fast_list.append(np.asarray(sfs_fast))
        xAxis_fast_list.append(np.asarray(xAxis_fast))

    plot4me(iters_list, energy_list, text='Energy-Iteration')
    plot4me(xAxis_fast_list, sfs_fast_list, text='operator-site')



#plot(dataname='run/L100.log', L=100)
#plot(dataname='run/L20_estimate.log', L=20, observables=True)

present(Ls=[10, 20, 30, 40], path='results/Gd')
