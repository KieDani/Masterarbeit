# Load the data from the .log file
import json
import matplotlib.pyplot as plt



def plot(dataname, L):
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
        for i in range(35):
            sum += array[-i+0]
        return sum/35.

    def getsf(i):
        sf = list()
        for iteration in data["Output"]:
            sf.append(iteration['Ferro_correlation_function' + str(i)]["Mean"])
        return calcMean(sf)

    for i in range(1, length):
        sfs_fast.append(getsf(i))
        xAxis_fast.append(i)

    plt.plot(iters, energy)
    plt.show()


    plt.plot(xAxis_fast, sfs_fast)
    plt.show()



#plot(dataname='run/L100.log', L=100)
plot(dataname='run/L10.log', L=10)
