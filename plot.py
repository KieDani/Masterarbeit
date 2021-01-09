# Load the data from the .log file
import json
import matplotlib.pyplot as plt
import numpy as np
import helping_functions as functions
from multiprocessing import Pool



def plot(dataname, L, observables=True, symmetric_operator = False, periodic=False, transformed_or_original = 'transformed'):
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
        length = np.minimum(15, len(array))
        sum = 0.
        for i in range(length):
            sum += array[-i+0]
        return sum / float(length)

    def getsf(i):
        sf = list()
        for iteration in data["Output"]:
            if(symmetric_operator == True):
                try:
                    sf.append(iteration['Symmetric_Ferro_correlation_function' + str(i)]["Mean"])
                except:
                    print('Symmetric operator is plotted. If I do not plot old data, I probably made a mistake!')
                    sf.append(iteration['Ferro_correlation_function' + str(i)]["Mean"])
            else:
                sf.append(iteration['Ferro_correlation_function' + str(i)]["Mean"])
        return calcMean(sf)

    plt.plot(iters, energy)
    if(periodic == False):
        tmp = [None, None, -1.999, -3.000, -4.646, -5.830, -7.370, -8.635, -10.125, -11.433, -12.895, -14.230, -15.674, -17.028, -18.459, -19.827, -21.250, -22.626]
    else:
        if(transformed_or_original == 'transformed'):
            #energy of transformed hamiltonian
            tmp = [None, None, None, None, -6.000, -7.096, -8.617, -9.863, -11.337, -12.647, -14.094, -15.438, -16.870, -18.234, -19.655, -21.032, -22.447, -23.832 ]
        else:
            #energy of the normal hamiltonian
            tmp = [None, None, None, None, -5.999, -6.531, -8.617, -9.572, -11.337, -12.480, -14.094, -15.337, -16.870, -18.170, -19.655, -20.991, -22.447 ]
    if (L < len(tmp)):
        factor = tmp[L]
    else:
        factor = (L - 1) * (-1.4)
    expected_energy = np.ones_like(np.asarray(iters)) * factor
    print(dataname + '; (E_exact - E)/E_exact = ' + str((factor - np.mean(energy[-int(1./3 * len(energy)):])) / factor))
    plt.plot(iters, expected_energy, color='red')
    plt.title(dataname)
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.show()

    if(observables == True):
        if(symmetric_operator == True):
            for i in range(1, int(L / 2.)):
                sfs_fast.append(getsf(2*i))
                xAxis_fast.append(2*i)
                if(periodic == False):
                    dataname_operator = ''.join(('run/exact_symmetricOperator_', transformed_or_original , '/L', str(L), '_exact.csv'))
                else:
                    dataname_operator = ''.join(('run/exact_periodic_symmetricOperator_', transformed_or_original , '/L', str(L), '_exact.csv'))
        else:
            for i in range(1, length):
                sfs_fast.append(getsf(i))
                xAxis_fast.append(i)
                if(periodic == False):
                    dataname_operator = ''.join(('run/exact_', transformed_or_original , '/L', str(L), '_exact.csv'))
                else:
                    dataname_operator = ''.join(('run/exact_periodic_', transformed_or_original , '/L', str(L), '_exact.csv'))

        plt.plot(xAxis_fast, sfs_fast)
        try:
            operator = -1 * np.loadtxt(dataname_operator)
            if(transformed_or_original == 'transformed'): operator = operator * -1
            if(symmetric_operator == True):
                x_operator = np.arange(2, 2*len(operator)+1, 2)
            else:
                x_operator = np.arange(1, len(operator)+1)
        except:
            print(dataname_operator)
            operator = 0.374 * np.ones(len(xAxis_fast))
            x_operator = xAxis_fast
        plt.plot(x_operator, operator, color='red')
        plt.title(dataname)
        plt.show()


def compare_original_transformed(L, periodic=False):
    if(periodic==True):
        dataname = ''.join(('run/exact_periodic_original/L', str(L), '_exact.csv'))
        dataname2 = ''.join(('run/exact_periodic_transformed/L', str(L), '_exact.csv'))
    else:
        dataname = ''.join(('run/exact_original/L', str(L), '_exact.csv'))
        dataname2 = ''.join(('run/exact_transformed/L', str(L), '_exact.csv'))
    try:
        operator_orig = -1 * np.loadtxt(dataname)
        x_operator_orig = np.arange(1, len(operator_orig) + 1)
        operator_trans = +1 * np.loadtxt(dataname2)
        x_operator_trans = np.arange(1, len(operator_trans) + 1)
        operator_inf = 0.374 * np.ones(len(x_operator_orig))
    except:
        print('L is too large')
    plt.plot(x_operator_orig, operator_orig, color='green', label='original Hamiltonian')
    plt.plot(x_operator_trans, operator_trans, color='blue', label='transformed Hamiltonian')
    plt.plot(x_operator_orig, operator_inf, color='red', label='expected infinity value')
    plt.title('Compare observable for periodic lattice')
    plt.xlabel('distance between sites')
    plt.ylabel('Observable')
    plt.legend()
    plt.show()


# Ls should be an array with 4 Elements
def present(Ls, path):

    #a = x data, b = f(x), c = expected energy
    def plot4me(a, b, c, text='Energy-Iterations'):
        plt.subplot(221)  # sublot(Anzahl Zeilen Anzahl Spalten Bild Nummer)
        plt.plot(a[0], b[0])
        if(c != None): plt.plot(a[0], c[0])
        plt.title(''.join((text, ' ', str(Ls[0]))))
        plt.subplot(222)  # sublot(Anzahl Zeilen Anzahl Spalten Bild Nummer)
        plt.plot(a[1], b[1])
        if (c != None): plt.plot(a[1], c[1])
        plt.title(''.join((text, ' ', str(Ls[1]))))
        plt.subplot(223)  # sublot(Anzahl Zeilen Anzahl Spalten Bild Nummer)
        plt.plot(a[2], b[2])
        if (c != None): plt.plot(a[2], c[2])
        plt.title(''.join((text, ' ', str(Ls[2]))))
        plt.subplot(224)  # sublot(Anzahl Zeilen Anzahl Spalten Bild Nummer)
        plt.plot(a[3], b[3])
        if (c != None): plt.plot(a[3], c[3])
        plt.title(''.join((text, ' ', str(Ls[3]))))
        plt.show()

    iters_list = []
    energy_list = []
    sfs_fast_list = []
    xAxis_fast_list = []
    expected_energy_list = []
    for j in range(0, len(Ls)):
        l = Ls[j]
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
            length = np.minimum(15, len(array))
            print(length)
            sum = 0.
            for i in range(length):
                sum += array[-i + 0]
            return sum / float(length)

        def getsf(i):
            sf = list()
            for iteration in data_observ["Output"]:
                sf.append(iteration['Ferro_correlation_function' + str(i)]["Mean"])
            return calcMean(sf)

        for i in range(1, length):
            sfs_fast.append(getsf(i))
            xAxis_fast.append(i)

        tmp = [None, None, -1.999, -3.000, -4.646, -5.830, -7.370, -8.635, -10.125, -11.433, -12.895, -14.230, -15.674, -17.028, -18.459, -19.827, -21.250, -22.626]
        if(l < len(tmp)):
            factor = tmp[l]
        else:
            factor = (l-1) * (-1.4)
        expected_energy = np.ones_like(np.asarray(iters)) * factor

        expected_energy_list.append(expected_energy)
        iters_list.append(np.asarray(iters))
        energy_list.append(np.asarray(energy))
        sfs_fast_list.append(np.asarray(sfs_fast))
        xAxis_fast_list.append(np.asarray(xAxis_fast))

    plot4me(iters_list, energy_list, expected_energy_list, text='Energy-Iteration')
    plot4me(xAxis_fast_list, sfs_fast_list, None, text='operator-site')



def plot_startingpoints(dataname, L, fast=True):
    data = json.load(open(dataname))
    # Extract the relevant information

    iters = []
    energy = []

    for iteration in data["Output"]:
        iters.append(iteration["Iteration"])
        energy.append(iteration["Energy"]["Mean"])

    def calcMean(array):
        length = np.minimum(15, len(array))
        sum = 0.
        for i in range(length):
            sum += array[-i + 0]
        return sum / float(length)

    def getsf(j, k):
        sf = list()
        for iteration in data["Output"]:
            sf.append(iteration[''.join((str(j), 'Ferro_correlation_function', str(k - j)))]["Mean"])
        return calcMean(sf)

    plt.plot(iters, energy)
    tmp = [None, None, -1.999, -3.000, -4.646, -5.830, -7.370, -8.635, -10.125, -11.433, -12.895, -14.230, -15.674, -17.028, -18.459, -19.827, -21.250, -22.626]
    if (L < len(tmp)):
        factor = tmp[L]
    else:
        factor = (L - 1) * (-1.4)
    expected_energy = np.ones_like(np.asarray(iters)) * factor
    plt.plot(iters, expected_energy, color = 'red')
    #print(energy +np.ones(len(iters)) * (L-1) * 1.4)
    plt.title('Energy-iteration')
    plt.show()

    colors = ['black', 'brown', 'grey', 'green', 'blue', 'orange', 'yellow']
    for start, j in enumerate([1, 2, 3, int(L/4.), int(L/2.)]):
        sfs_fast = list()
        xAxis_fast = list()
        if(fast == True):
            max_range = L
        else:
            max_range = np.minimum(j + 8, L)
        for k in range(j + 1, max_range):
            sfs_fast.append(getsf(j, k))
            xAxis_fast.append(k-j)

        plt.plot(xAxis_fast, sfs_fast, color= colors[start], label=''.join(('startingpoint: ', str(j))))
        plt.plot(xAxis_fast, 0.374 * np.ones(len(xAxis_fast)), color = 'red')
        plt.legend()
    plt.title('operator-distance')
    plt.show()


def plot_Sr(path, L):
    def calcMean(array):
        length = np.minimum(15, len(array))
        sum = 0.
        for i in range(length):
            sum += array[-i + 0]
        return sum / float(length)

    def getsf(i, data):
        sf = list()
        for iteration in data["Output"]:
            sf.append(iteration['Ferro_correlation_function' + str(i)]["Mean"])
        return calcMean(sf)

    iterations = list()
    energies = list()
    strincorrs = list()
    xes = list()
    Sr = [0.01, 0.1,  1, 10, None]
    for i, sr in enumerate(Sr):
        sr_string = '_'.join((str(sr).split('.')))
        dataname = ''.join((path, 'Sr', sr_string, 'L', str(L), '.log'))
        dataname2 = ''.join((path, 'Sr', sr_string, 'L', str(L), '_estimate.log'))
        data = json.load(open(dataname))
        data2 = json.load(open(dataname2))
        iters = []
        energy = []
        for iteration in data["Output"]:
            iters.append(iteration["Iteration"])
            energy.append(iteration["Energy"]["Mean"])
        iterations.append(np.asarray(iters))
        energies.append(np.asarray(energy))

        sfs_fast = list()
        xAxis_fast = list()
        for i in range(1, int(L / 2.)):
            sfs_fast.append(getsf(2 * i, data2))
            xAxis_fast.append(2 * i)
        strincorrs.append(np.asarray(sfs_fast))
        xes.append(np.asarray(xAxis_fast))

    fig, axes = plt.subplots(2, 3)
    for i in range(0, len(Sr)):
        fig.suptitle('Energy - Iterations')
        axes[int(i / 3), i % 3].plot(iterations[i], energies[i])
        tmp = [None, None, -1.999, -3.000, -4.646, -5.830, -7.370, -8.635, -10.125, -11.433, -12.895, -14.230, -15.674, -17.028, -18.459, -19.827, -21.250, -22.626]
        if (L < len(tmp)):
            factor = tmp[L]
        else:
            factor = (L - 1) * (-1.4)
        expected_energy = np.ones_like(np.asarray(iters)) * factor
        axes[int(i / 3), i % 3].plot(iters, expected_energy, color='red')
        axes[int(i / 3), i % 3].set_title(''.join(('Sr', str(Sr[i]), 'L', str(L), '.log')))
    plt.show()
    fig, axes = plt.subplots(2, 3)
    for i in range(0, len(Sr)):
        fig.suptitle('Stringcorrelation - distance')
        print(strincorrs[i])
        axes[int(i / 3), i % 3].plot(xes[i], strincorrs[i])
        try:
            dataname = ''.join(('run/exact_original/L', str(L), '_exact.csv'))
            operator = -1 * np.loadtxt(dataname)
            x_operator = np.arange(1, len(operator)+1)
        except:
            operator = 0.374 * np.ones(len(xAxis_fast))
            x_operator = xAxis_fast
        axes[int(i / 3), i % 3].plot(x_operator, operator, color='red')
        axes[int(i / 3), i % 3].set_title(''.join(('Sr', str(Sr[i]), 'L', str(L), '.log')))
    plt.show()


def plot_operator_both_sides(dataname, L):
    data = json.load(open(dataname))
    # Extract the relevant information

    iters = []
    energy = []

    for iteration in data["Output"]:
        iters.append(iteration["Iteration"])
        energy.append(iteration["Energy"]["Mean"])

    def calcMean(array):
        length = np.minimum(15, len(array))
        sum = 0.
        for i in range(length):
            sum += array[-i + 0]
        return sum / float(length)

    def getsf(j, k, mirrored = ''):
        sf = list()
        for iteration in data["Output"]:
            sf.append(iteration[''.join(('Ferro_correlation_function', mirrored, str(k - j)))]["Mean"])
        return calcMean(sf)

    plt.plot(iters, energy)
    plt.plot(iters, -np.ones(len(iters)) * (L-1) * 1.4, color = 'red')
    print(energy +np.ones(len(iters)) * (L-1) * 1.4)
    plt.title('Energy-iteration')
    plt.show()

    colors = ['black', 'blue']
    for index, mirrored in enumerate(['', '_mirrored']):
        sfs_fast = list()
        xAxis_fast = list()
        max_range = L
        for i in range(1, int(L/2.)):
            sfs_fast.append(getsf(int(L/2.), int(L/2.) + i, mirrored=mirrored))
            xAxis_fast.append(i)

        plt.plot(xAxis_fast, sfs_fast, color= colors[index], label=''.join(('operator', mirrored)))
        plt.plot(xAxis_fast, 0.374 * np.ones(len(xAxis_fast)), color = 'red')
        plt.legend()
    plt.title('operator-distance')
    plt.show()



#plot(dataname='run/L100.log', L=100)
#plot(dataname='run/L20_estimate.log', L=20, observables=True)

#present(Ls=[6, 10, 15, 20], path='results/Sr')

#plot_startingpoints('run/startingpoint_superpower/L30_estimate.log', 30, fast=True)



#results operator both sides


#RBM
L=40
machine = '_RBM'
#plot(dataname='run/operator_both_sides'+ machine + '/L' + str(L) + '.log', L=L, observables=False)
#plot_operator_both_sides(dataname='run/operator_both_sides' + machine + '/L' + str(L) + '_estimate.log', L=L)

#SymRBM
L=40
machine = '_SymRBM'
#plot(dataname='run/operator_both_sides'+ machine + '/L' + str(L) + '.log', L=L, observables=False)
#plot_operator_both_sides(dataname='run/operator_both_sides' + machine + '/L' + str(L) + '_estimate.log', L=L)


#DeepFFNN
L=40
machine = '_DeepFFNN'
#plot(dataname='run/operator_both_sides'+ machine + '/L' + str(L) + '.log', L=L, observables=False)
#plot_operator_both_sides(dataname='run/operator_both_sides' + machine + '/L' + str(L) + '_estimate.log', L=L)



#results with symmetric operator


#RBM
L=40
machine = '_RBM'
#plot(dataname='run/symmetric_operator'+ machine + '/L' + str(L) + '.log', L=L, observables=False)
#plot('run/symmetric_operator'+ machine + '/L' + str(L) + '_estimate.log', L=L, symmetric_operator=True, observables=True)


#FFNN
L=45
machine = '_FFNN'
#plot(dataname='run/symmetric_operator'+ machine + '/L' + str(L) + '.log', L=L, observables=False)
#plot('run/symmetric_operator'+ machine + '/L' + str(L) + '_estimate.log', L=L, symmetric_operator=True, observables=True)



#DeepFFNN
L=60
machine = '_DeepFFNN'
#plot(dataname='run/symmetric_operator'+ machine + '/L' + str(L) + '.log', L=L, observables=False)
#plot('run/symmetric_operator'+ machine + '/L' + str(L) + '_estimate.log', L=L, symmetric_operator=True, observables=True)



#results test_sr


#Compare Sr RBM
#plot_Sr(path='run/test_sr/', L=40)

#Compare Sr FFNN
#plot_Sr(path='run/test_sr_ffnn/', L=40)
#plot_Sr(path='run/test_sr_FFNN/', L=12)


#plot('run/small_RBM/SrNoneL14_estimate.log', L = 14 ,symmetric_operator=False, observables=True)

#compare_original_transformed(L=16, periodic=True)

#plot('run/small_FFNN_periodic/SrNoneL8_estimate.log', L = 8 ,symmetric_operator=False, observables=True, periodic=True, transformed_or_original='original')

#plot('run/small_FFNN/SrNoneL30_estimate.log', L = 30 ,symmetric_operator=False, observables=True, periodic=False, transformed_or_original='transformed')




#plot('run/firstResults_FFNN/L32_estimate.log', L = 32 ,symmetric_operator=False, observables=True, periodic=False, transformed_or_original='transformed')

def wrapper(i):
    machine_names = ['JaxRBM', 'JaxSymmRBM', 'JaxFFNN', 'JaxDeepFFNN', 'JaxSymmFFNN', 'JaxUnaryFFNN', 'JaxConv3NN', 'JaxResFFNN', 'JaxResConvNN']
    plot('run/compareArchitectures/CPU/Iterations/' + machine_names[i] + '/L16.log', L = 16 ,symmetric_operator=False, observables=False, periodic=False, transformed_or_original='transformed')

# with Pool(8) as p:
#     p.map(wrapper, [0, 2, 3, 4, 5, 6, 7, 8])

