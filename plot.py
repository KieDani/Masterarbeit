"""Implementation of some functions to plot the results.

This project requires the following libraries:
netket, numpy, scipy, jax, jaxlib, networkx, torch, tqdm, matplotlib

This file contains the following functions:

    * plot
    * plotObservable
    * plotS_Z_squared
    * compare_original_transformed
    * plot_startingpoints
    * plot_Sr
    * plot_operator_both_sides
    * compareArchitectures
    * plotEnergyPerSize
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import csv



def plot(dataname, L, observables=True, symmetric_operator = False, periodic=False, transformed_or_original = 'transformed', title=None):
    """Function to plot the results of the calculations
        Multiple possibilities to plot the results

            Args:
                dataname (str) : the dataname (with the relative path)
                L (int) : Lattice size
                symmetric_operator (bool) : if the observable is measured symmetrically to the center
                periodic (bool) : if exact results of the periodic lattice are plotted
                transformed_or_original (str) : which hamiltonian is used. 'transformed' or 'original' or 'AKLT'
                title (str): title of the plot
                                                    """
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

    plt.plot(iters, energy, label='VMC energy')
    if(periodic == False):
        tmp = [None, None, -1.999, -3.000, -4.646, -5.830, -7.370, -8.635, -10.125, -11.433, -12.895, -14.230, -15.674, -17.028, -18.459, -19.827, -21.250, -22.626]
    else:
        if(transformed_or_original == 'transformed'):
            #energy of transformed hamiltonian
            tmp = [None, None, None, None, -6.000, -7.096, -8.617, -9.863, -11.337, -12.647, -14.094, -15.438, -16.870, -18.234, -19.655, -21.032, -22.447, -23.832 ]
        else:
            #energy of the normal hamiltonian
            tmp = [None, None, None, None, -5.999, -6.531, -8.617, -9.572, -11.337, -12.480, -14.094, -15.337, -16.870, -18.170, -19.655, -20.991, -22.447 ]
    if (L < len(tmp) and transformed_or_original != 'AKLT'):
        factor = tmp[L]
    elif(transformed_or_original == 'AKLT'):
        factor = -2./3 * (L - 1)
    else:
        factor = (L - 1) * (-1.4)
    expected_energy = np.ones_like(np.asarray(iters)) * factor
    print(dataname + '; (E_exact - E)/E_exact = ' + str((factor - np.mean(energy[-int(1./3 * len(energy)):])) / factor))
    plt.plot(iters, expected_energy, color='red', label='exact energy')
    if title==None:
        plt.title(dataname)
    else:
        plt.title(title, fontsize=14)
    plt.xlabel('Iteration', fontsize = 14)
    plt.ylabel('Energy', fontsize = 14)
    plt.legend()
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



def plotObservables(dataname, L, observable='FerroCorr', title = None, hamiltonian = 'Heisenberg', yLabel = None):
    """Function to plot the results of the function measureObservables().
        The csv file is loaded and evaluated.

            Args:
                dataname (str) : the dataname (including the relative path)
                L (int) : Lattice size
                observable (str): allowed inputs are 'FerroCorr', 'SymmFerroCorr' and 'StringCorr'
                title (str) : Title of the plot
                hamiltonian (str): allowed inputs are 'Heisenberg' or 'AKLT'
                yLabel (str): ylabel of the plot
                                                        """
    #observable 1 at position 0, etc
    numbers = np.zeros(L-1, dtype=np.int32)
    values = np.zeros(L-1, dtype=np.float64)

    with open(dataname) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            for i in range(0, L-1):
                if observable == 'FerroCorr':
                    name_observable = 'Ferro_correlation_function'
                elif observable == 'SymmFerroCorr':
                    name_observable = 'Symmetric_Ferro_correlation_function'
                elif observable == 'StringCorr':
                    name_observable = 'String_correlation_function'
                else:
                    name_observable = None
                    print('wrong input for parameter observable')
                if row[0] == ''.join((name_observable, str(i+1))):
                    value = row[1].split('+')[0]
                    if value[-1] == 'e':
                        value = ''.join((value, row[1].split('+')[1]))
                    value = float(value)
                    values[i] += value
                    if observable == 'SymmFerroCorr':
                        values[i+1] += value
                        numbers[i+1] += 1

                    numbers[i] += 1
        print(values)
        print(numbers)
        print(values/numbers)
        plt.plot(range(1, L), values/numbers, label='VMC value')
        dataname_observable = ''.join(('run/exact_', 'transformed', '/L', str(L), '_exact.csv'))
        if(hamiltonian == 'AKLT'):
            operator = 4./9 * np.ones(len(values))
            x_operator = range(1, L)
        else:
            try:
                operator = 1 * np.loadtxt(dataname_observable)
                x_operator = np.arange(1, len(operator) + 1)
            except:
                try:
                    print('KuhMachtMuh')
                    if(observable == 'SymmFerroCorr'):
                        dataname_observ = ''.join(('run/DMRG/DMRG_symm_', str(L), '.csv'))
                    else:
                        dataname_observ = ''.join(('run/DMRG/DMRG_', str(L), '.csv'))
                        print(''.join(('run/DMRG/DMRG_', str(L), '.csv')))
                    observables = list()
                    with open(dataname_observ) as csvfile:
                        spamreader = csv.reader(csvfile)
                        for row in spamreader:
                            observables.append(np.abs(float(row[0])))
                            if(observable == 'SymmFerroCorr'):
                                observables.append(float(row[0]))
                    operator = np.asarray(observables)
                    print(operator)
                    x_operator = np.arange(1, len(operator) + 1)
                except:
                    print('MuhMachtKuh')
                    print(dataname_observable)
                    operator = 0.374 * np.ones(len(values))
                    x_operator = range(1, L)
        plt.plot(x_operator, operator, color='red', label='exact value')
        plt.xlabel('Site distance', fontsize = 14)
        if(observable == 'FerroCorr' and yLabel == None):
            plt.ylabel('Ferromagnetic order parameter', fontsize=14)
        elif(yLabel == None):
            plt.ylabel('String order parameter', fontsize=14)
        else:
            plt.ylabel(yLabel, fontsize=14)
        if title == None:
            plt.title(dataname)
        else:
            plt.title(title, fontsize = 14)
        plt.legend()
        plt.show()


def plotS_Z_squared(dataname, L, title=None):
    """Function to plot the results of the function measureObservables() for S_Z_squared.
            The csv file is loaded and evaluated.

                Args:
                    dataname (str) : the dataname (including the relative path)
                    L (int) : Lattice size
                    title (str) : Title of the plot
                                                            """
    # observable 0 at position 0, etc
    numbers = np.zeros(L, dtype=np.int32)
    values = np.zeros(L, dtype=np.float64)

    with open(dataname) as csvfile:
        spamreader = csv.reader(csvfile)
        name_observable = 'S_Z_squared'
        for row in spamreader:
            for i in range(0, L):
                if row[0] == ''.join((name_observable, str(i))):
                    value = row[1].split('+')[0]
                    if value[-1] == 'e':
                        value = ''.join((value, row[1].split('+')[1]))
                    value = float(value)
                    values[i] += value
                    numbers[i] += 1
        print(values)
        print(numbers)
        print(values/numbers)

        plt.plot(range(0, L), values/numbers)
        plt.title(title, fontsize=14)
        plt.xlabel(r'lattice site $i$', fontsize=14)
        plt.ylabel(r'$< { S_i^{(z)} } ^2 >$', fontsize=14)
        plt.show()

        number_nonzero = values/numbers
        number_nonzero = np.sum(number_nonzero)
        print('Number non-zero', number_nonzero)
        print('Number zero', L - number_nonzero)



def compare_original_transformed(L, periodic=False):
    """comparison of the exact results of the original and periodic heisenberg hamiltonian
    """
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



def plot_startingpoints(dataname, L, fast=True):
    """plots results of the observable for multiple starting points"""
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
    """compares multiple Sr values"""
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
    """plot of operator. One is starting at the center going to the left. The other is going to the right."""
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
        for i in range(1, int(L/2.)):
            sfs_fast.append(getsf(int(L/2.), int(L/2.) + i, mirrored=mirrored))
            xAxis_fast.append(i)

        plt.plot(xAxis_fast, sfs_fast, color= colors[index], label=''.join(('operator', mirrored)))
        plt.plot(xAxis_fast, 0.374 * np.ones(len(xAxis_fast)), color = 'red')
        plt.legend()
    plt.title('operator-distance')
    plt.show()


def compareArchitectures(machine_names, path, L):
    """Function to compare the results of different architectures.
        Mean energy, variance, mean time, and so on are evaluated.

            Args:
                machine_names (list) : list with machine names (str) as elements
                path (str) : path to the data folder
                L (int) : Lattice size
                                                        """
    for machine_name in machine_names:
        deviations_energy = list()
        times = list()
        for i in range(0, 5):
            try:
                dataname = ''.join((path, machine_name, '/', str(i), 'L', str(L), '.log'))
                data = json.load(open(dataname))
                iters = []
                energy = []

                for iteration in data["Output"]:
                    iters.append(iteration["Iteration"])
                    energy.append(iteration["Energy"]["Mean"])

                tmp = [None, None, -1.999, -3.000, -4.646, -5.830, -7.370, -8.635, -10.125, -11.433, -12.895, -14.230, -15.674,
                       -17.028, -18.459, -19.827, -21.250, -22.626]
                if (L < len(tmp)):
                    factor = tmp[L]
                else:
                    factor = (L - 1) * (-1.401484)
                deviation_energy = (factor - np.mean(energy[350-int(1./3*len(energy)):350])) / factor
                deviations_energy.append(deviation_energy)
                # Time data is created at the end of the simulation -> There might be no .time data yet
                try:
                    with open(''.join((dataname.split('.')[0], '.time')), 'r') as reader:
                        time = reader.read()
                        #print(time)
                        times.append(float(time))
                except:
                    pass
            except:
                #Results not yet finished
                pass
        #print(deviations_energy)
        mean = np.mean(deviations_energy)
        median = np.median(deviations_energy)
        variance = np.var(deviations_energy, dtype=np.float64)
        standard_deviation = np.sqrt(variance)
        minimum = np.amin(deviations_energy)
        maximum = np.amax(deviations_energy)
        meantime = np.mean(times)
        #mintime  = np.amin(times)
        print(''.join((machine_name + '; mean=', str(mean), ' variance=', str(variance), ' standard deviation=', str(standard_deviation), ' median=', str(median), ' minimum=', str(minimum), ' maximum=', str(maximum), ' time=', str(meantime))))



def plotEnergyPerSize():
    """Function to plot the scaling of the energy of the Haldane chain.
            Scaling per lattice site and scaling per bond are compared.

                Args:

                                                            """
    lanczosEnergy = np.asarray([-1.999, -3.000, -4.646, -5.830, -7.370, -8.635, -10.125, -11.433, -12.895, -14.230, -15.674, -17.028, -18.459, -19.827, -21.250, -22.626])
    Ls = np.asarray(range(2, len(lanczosEnergy) + 2))
    DMRG_Energy = lanczosEnergy / Ls
    adjusted_Energy = lanczosEnergy / (Ls - np.ones_like(Ls))
    energy_per_site = -1.401 * np.ones_like(Ls)
    plt.plot(Ls, energy_per_site, color='red', label='exact value')
    plt.plot(Ls, DMRG_Energy, color='blue', label='E/N')
    plt.plot(Ls, adjusted_Energy, color='black', label='E/(N-1)')
    plt.title('Scaling of the ground state energy', fontsize=14)
    plt.xlabel('Lattice size', fontsize=14)
    plt.ylabel('Energy', fontsize=14)
    plt.legend()
    plt.show()
    print(DMRG_Energy)
    print(adjusted_Energy)


def compareDMRG(dataname, L, mode=None):
    numbers = np.zeros(L - 1, dtype=np.int32)
    values = np.zeros(L - 1, dtype=np.float64)

    with open(dataname) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            for i in range(0, L - 1):
                name_observable = 'Ferro_correlation_function'
                if row[0] == ''.join((name_observable, str(i + 1))):
                    value = row[1].split('+')[0]
                    if value[-1] == 'e':
                        value = ''.join((value, row[1].split('+')[1]))
                    value = float(value)
                    values[i] += value
                    numbers[i] += 1
        print(values)
        print(numbers)
        print(values / numbers)
        plt.plot(range(1, L), values / numbers, label='VMC value')

        if(mode == 'symmetric'):
            dataname_observ = ''.join(('run/DMRG/DMRG_symm_', str(L), '.csv'))
        else:
            dataname_observ = ''.join(('run/DMRG/DMRG_', str(L), '.csv'))

        observables = list()
        with open(dataname_observ) as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                observables.append(np.abs(float(row[0])))
                if(mode == 'symmetric'):
                    observables.append(float(row[0]))
        print('observables: ', observables)
        plt.plot(observables, color='red', label='DMRG value')

        dataname_exact = ''.join(('run/exact_', 'transformed', '/L', str(L), '_exact.csv'))
        try:
            operator = 1 * np.loadtxt(dataname_exact)
            x_operator = np.arange(1, len(operator) + 1)
        except:
            print(dataname_exact)
            operator = 0.374 * np.ones(len(values))
            x_operator = range(1, L)
        plt.plot(x_operator, operator, color='green', label='exact value')

        if mode != None:
            plt.title(mode)

        plt.legend()
        plt.show()



def compareLatticeSize(L):

    # Get the energy
    average_energies = []
    for index in range(1, 9):
        dataname_energy = 'run/CompareSymmObserv/L' + str(L) + '_load' + str(index) + '.log'
        data = json.load(open(dataname_energy))
        iters = []
        energy = []
        for iteration in data["Output"]:
            iters.append(iteration["Iteration"])
            energy.append(iteration["Energy"]["Mean"])
        def calcMean(array):
            length = np.minimum(25, len(array))
            sum = 0.
            for i in range(length):
                sum += array[-i+0]
            return sum / float(length)
        average_energies.append(calcMean(energy))
    average_energy = np.mean(average_energies)
    average_energy_std = np.std(average_energies)

    with open('run/DMRG/DMRG_Energy' + str(L) + '.csv') as csvfile:
        spamreader = csv.reader(csvfile)
        dmrg_energy = next(spamreader)
        dmrg_energy = float(dmrg_energy[0].split('+')[0] + dmrg_energy[0].split('+')[1])

    print('VMC-Energy: ', average_energy, ' +- ', average_energy_std)
    print('DMRG-Energy: ', dmrg_energy)
    print('(E_dmrg - E)/E_dmrg = ', (dmrg_energy - average_energy) / dmrg_energy)



    #Get the observables
    numbers = np.zeros(L - 1, dtype=np.int32)
    values = np.zeros(L - 1, dtype=np.float64)
    for index in range(1, 9):
        dataname = 'run/CompareSymmObserv/L' + str(L) + '_load' + str(index) + '_observables.csv'
        with open(dataname) as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                for i in range(0, L - 1):
                    name_observable = 'Ferro_correlation_function'
                    if row[0] == ''.join((name_observable, str(i + 1))):
                        value = row[1].split('+')[0]
                        if value[-1] == 'e':
                            value = ''.join((value, row[1].split('+')[1]))
                        value = float(value)
                        values[i] += value
                        numbers[i] += 1
    plt.plot(range(1, L), values / numbers, label='VMC value')

    try:
        dataname_observ = ''.join(('run/DMRG/DMRG_', str(L), '.csv'))
        print(''.join(('run/DMRG/DMRG_', str(L), '.csv')))
        observables = list()
        with open(dataname_observ) as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                observables.append(np.abs(float(row[0])))
        operator = np.asarray(observables)
        x_operator = np.arange(1, len(operator) + 1)
        plt.plot(x_operator, operator, color='red', label='DMRG value')
    except:
        print('not possible to plot DMRG data')

    plt.xlabel('Site distance', fontsize=14)
    plt.ylabel('String order parameter', fontsize=14)
    plt.title('L = ' + str(L))
    plt.legend()
    plt.show()



def compareLatticeSizes():
    energies_L = list()
    dmrgenergies_L = list()
    orderparameter_L = list()
    dmrgorderparameter_L = list()
    for l in [30, 40, 50, 60]:
        average_energies = []
        for index in range(1, 9):
            dataname_energy = 'run/CompareSymmObserv/L' + str(l) + '_load' + str(index) + '.log'
            data = json.load(open(dataname_energy))
            iters = []
            energy = []
            for iteration in data["Output"]:
                iters.append(iteration["Iteration"])
                energy.append(iteration["Energy"]["Mean"])

            def calcMean(array):
                length = np.minimum(25, len(array))
                sum = 0.
                for i in range(length):
                    sum += array[-i + 0]
                return sum / float(length)

            average_energies.append(calcMean(energy))
        average_energy = np.mean(average_energies)
        average_energy_std = np.std(average_energies)
        energies_L.append(average_energy)

        with open('run/DMRG/DMRG_Energy' + str(l) + '.csv') as csvfile:
            spamreader = csv.reader(csvfile)
            dmrg_energy = next(spamreader)
            dmrg_energy = float(dmrg_energy[0].split('+')[0] + dmrg_energy[0].split('+')[1])
        dmrgenergies_L.append(dmrg_energy)

        dataname_observ = ''.join(('run/DMRG/DMRG_', str(l), '.csv'))
        print(''.join(('run/DMRG/DMRG_', str(l), '.csv')))
        observables = list()
        with open(dataname_observ) as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                observables.append(np.abs(float(row[0])))
        operator = np.asarray(observables)
        dmrgorderparameter_L.append(operator[int(l/2-1)])

        values = 0.
        numbers = 0.
        for index in range(1, 9):
            dataname = 'run/CompareSymmObserv/L' + str(l) + '_load' + str(index) + '_observables.csv'
            with open(dataname) as csvfile:
                spamreader = csv.reader(csvfile)
                for row in spamreader:
                    name_observable = 'Ferro_correlation_function'
                    if row[0] == ''.join((name_observable, str(int(l/2)))):
                        value = row[1].split('+')[0]
                        if value[-1] == 'e':
                            value = ''.join((value, row[1].split('+')[1]))
                        value = float(value)
                        values += value
                        numbers += 1
        orderparameter_L.append(values/numbers)

    print('VMC Energies:', energies_L)
    print('DMRG Energies:', dmrgenergies_L)
    print('(E_DMRG-E_VMC)/(E_VMC):', (np.array(dmrgenergies_L) - np.array(energies_L))/np.array(dmrgenergies_L))
    print('(E_DMRG-E_VMC):', np.array(dmrgenergies_L) - np.array(energies_L))
    print('VMC Order parameter:', orderparameter_L)
    print('DMRG Order parameter:', dmrgorderparameter_L)
    print('(O_DMRG-O_VMC)/(O_VMC):', abs(np.array(dmrgorderparameter_L) - np.array(orderparameter_L)) / np.array(dmrgorderparameter_L))
    print('(O_DMRG-O_VMC):', abs(np.array(dmrgorderparameter_L) - np.array(orderparameter_L)))






def compareNetworkSize(alpha):
    L=40
    # Get the energy
    average_energies = []
    for index in range(1, 9):
        dataname_energy = 'run/CompareSizes/L' + str(L) + 'a' + str(alpha) + '_load' + str(index) + '.log'
        data = json.load(open(dataname_energy))
        iters = []
        energy = []
        for iteration in data["Output"]:
            iters.append(iteration["Iteration"])
            energy.append(iteration["Energy"]["Mean"])
        def calcMean(array):
            length = np.minimum(25, len(array))
            sum = 0.
            for i in range(length):
                sum += array[-i+0]
            return sum / float(length)
        average_energies.append(calcMean(energy))
    average_energy = np.mean(average_energies)
    average_energy_std = np.std(average_energies)

    with open('run/DMRG/DMRG_Energy' + str(L) + '.csv') as csvfile:
        spamreader = csv.reader(csvfile)
        dmrg_energy = next(spamreader)
        dmrg_energy = float(dmrg_energy[0].split('+')[0] + dmrg_energy[0].split('+')[1])

    print('VMC-Energy: ', average_energy, ' +- ', average_energy_std)
    print('DMRG-Energy: ', dmrg_energy)
    print('(E_dmrg - E)/E_dmrg = ', (dmrg_energy - average_energy) / dmrg_energy)



    #Get the observables
    numbers = np.zeros(L - 1, dtype=np.int32)
    values = np.zeros(L - 1, dtype=np.float64)
    for index in range(1, 9):
        dataname = 'run/CompareSizes/L' + str(L) + 'a' + str(alpha) + '_load' + str(index) + '_observables.csv'
        with open(dataname) as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                for i in range(0, L - 1):
                    name_observable = 'Ferro_correlation_function'
                    if row[0] == ''.join((name_observable, str(i + 1))):
                        value = row[1].split('+')[0]
                        if value[-1] == 'e':
                            value = ''.join((value, row[1].split('+')[1]))
                        value = float(value)
                        values[i] += value
                        numbers[i] += 1
    plt.plot(range(1, L), values / numbers, label='VMC value')

    try:
        dataname_observ = ''.join(('run/DMRG/DMRG_', str(L), '.csv'))
        print(''.join(('run/DMRG/DMRG_', str(L), '.csv')))
        observables = list()
        with open(dataname_observ) as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                observables.append(np.abs(float(row[0])))
        operator = np.asarray(observables)
        x_operator = np.arange(1, len(operator) + 1)
        plt.plot(x_operator, operator, color='red', label='DMRG value')
    except:
        print('not possible to plot DMRG data')

    plt.xlabel('Site distance', fontsize=14)
    plt.ylabel('String order parameter', fontsize=14)
    plt.title('alpha = ' + str(alpha))
    plt.legend()
    plt.show()



def compareNetworkSizes():
    L = 40
    energies_alpha = list()
    dmrgenergies_alpha = list()
    orderparameter_alpha = list()
    dmrgorderparameter_alpha = list()
    for alpha in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
        average_energies = []
        for index in range(1, 9):
            dataname_energy = 'run/CompareSizes/L' + str(L) + 'a' + str(alpha) + '_load' + str(index) + '.log'
            data = json.load(open(dataname_energy))
            iters = []
            energy = []
            for iteration in data["Output"]:
                iters.append(iteration["Iteration"])
                energy.append(iteration["Energy"]["Mean"])

            def calcMean(array):
                length = np.minimum(25, len(array))
                sum = 0.
                for i in range(length):
                    sum += array[-i + 0]
                return sum / float(length)

            average_energies.append(calcMean(energy))
        average_energy = np.mean(average_energies)
        average_energy_std = np.std(average_energies)
        energies_alpha.append(average_energy)

        with open('run/DMRG/DMRG_Energy' + str(L) + '.csv') as csvfile:
            spamreader = csv.reader(csvfile)
            dmrg_energy = next(spamreader)
            dmrg_energy = float(dmrg_energy[0].split('+')[0] + dmrg_energy[0].split('+')[1])
        dmrgenergies_alpha.append(dmrg_energy)

        dataname_observ = ''.join(('run/DMRG/DMRG_', str(L), '.csv'))
        print(''.join(('run/DMRG/DMRG_', str(L), '.csv')))
        observables = list()
        with open(dataname_observ) as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                observables.append(np.abs(float(row[0])))
        operator = np.asarray(observables)
        dmrgorderparameter_alpha.append(operator[int(L/2-1)])

        values = 0.
        numbers = 0.
        for index in range(1, 9):
            dataname = 'run/CompareSizes/L' + str(L) + 'a' + str(alpha) + '_load' + str(index) + '_observables.csv'
            with open(dataname) as csvfile:
                spamreader = csv.reader(csvfile)
                for row in spamreader:
                    name_observable = 'Ferro_correlation_function'
                    if row[0] == ''.join((name_observable, str(int(L/2)))):
                        value = row[1].split('+')[0]
                        if value[-1] == 'e':
                            value = ''.join((value, row[1].split('+')[1]))
                        value = float(value)
                        values += value
                        numbers += 1
        orderparameter_alpha.append(values/numbers)

    print('VMC Energies:', energies_alpha)
    print('DMRG Energies:', dmrgenergies_alpha)
    print('(E_DMRG-E_VMC)/(E_VMC):', (np.array(dmrgenergies_alpha) - np.array(energies_alpha))/np.array(dmrgenergies_alpha))
    print('(E_DMRG-E_VMC):',np.array(dmrgenergies_alpha) - np.array(energies_alpha))
    print('VMC Order parameter:', orderparameter_alpha)
    print('DMRG Order parameter:', dmrgorderparameter_alpha)
    print('(O_DMRG-O_VMC)/(O_VMC):', abs(np.array(dmrgorderparameter_alpha) - np.array(orderparameter_alpha)) / np.array(dmrgorderparameter_alpha))
    print('(O_DMRG-O_VMC):',abs(np.array(dmrgorderparameter_alpha) - np.array(orderparameter_alpha)))
    plt.plot(abs(np.array(dmrgorderparameter_alpha) - np.array(orderparameter_alpha)))
    plt.show()



#Show that the original Heisenberg model and AKLT model can not be solved properly
#plot('results/problems/FFNN/L12.log', L = 12, symmetric_operator=False, observables=False, periodic=False, transformed_or_original='original', title='VMC energy of the Haldane chain (N=12)')
#plotObservables('results/problems/FFNN/L12_observables.csv', 12, observable='StringCorr', title='String order parameter for the Haldane chain (N=12)')
#plotObservables('results/problems/FFNN/L12_observables.csv', 12, observable='FerroCorr', title='Ferromagnetic order parameter for the Haldane chain (N=12)')

#plot('results/problemsAKLT/FFNN/L12.log', L=12, symmetric_operator=False, observables=False, periodic=False, transformed_or_original='AKLT', title='VMC energy of the AKLT model (N=12)')
#plotObservables('results/problemsAKLT/FFNN/L12_observables.csv', 12, observable='StringCorr', title='String order parameter for the AKLT model (N=12)')
#plotObservables('results/problemsAKLT/FFNN/L12_observables.csv', 12, observable='FerroCorr', title='Ferromagnetic order parameter for the AKLT model (N=12)')



#Results for transformed hamiltonian
#plot('results/transformedHamiltonian/L16.log', L=16, symmetric_operator=False, observables=False, periodic=False, transformed_or_original='transformed', title ='VMC energy of the transformed Haldane chain (N=16)')
#plotObservables('results/transformedHamiltonian/L16_observables.csv', 16, title='String order parameter for the transformed Haldane chain (N=16)', yLabel='String order parameter')
#plot('results/transformedHamiltonian/L60.log', L=60, symmetric_operator=False, observables=False, periodic=False, transformed_or_original='transformed', title='VMC energy of the transformed Haldane chain (N=60)')
#plotObservables('results/transformedHamiltonian/L60_observables.csv', 60, title='String order parameter for the transformed Haldane chain (N=60)', yLabel='String order parameter')
#plot('results/transformedHamiltonian/L80.log', L=80, symmetric_operator=False, observables=False, periodic=False, transformed_or_original='transformed', title='VMC energy of the transformed Haldane chain (N=80)')
#plotObservables('results/transformedHamiltonian/L80_observables.csv', 80, title='String order parameter for the transformed Haldane chain (N=80)', yLabel='String order parameter')
#plotObservables('results/transformedHamiltonian/L80_load_observables.csv', 80, title='String order parameter for the transformed Haldane chain (N=80)', yLabel='String order parameter')
#plot('results/transformedAKLT/FFNN/L12.log', L=12, transformed_or_original='AKLT', observables=False, periodic = False)
#plotObservables('results/transformedAKLT/FFNN/L12_observables.csv', L=12, hamiltonian='AKLT')

#plot('run/fifthResults/JaxFFNN/L16.log', L=16, symmetric_operator=False, observables=False, periodic=False, transformed_or_original='transformed', title ='VMC energy of the transformed Haldane chain (N=16)')
#plotObservables('run/fifthResults/JaxFFNN/L16_observables.csv', 16, title='String order parameter for the transformed Haldane chain (N=16)', yLabel='String order parameter')


#Comparison of architectures
#compareArchitectures(machine_names, path='run/compareArchitectures/CPU/Iterations/', L=16)
#machine_names = ['JaxResFFNN', 'JaxResConvNN', 'JaxDeepConvNN', 'JaxSymmFFNN', 'JaxDeepFFNN', 'JaxFFNN', 'JaxRBM']
#compareArchitectures(machine_names, path='results/compareArchitectures/', L=16)


#Test VBSSampler and InverseSampler
#plot('results/InverseSampler/FFNN/L12.log', L=12, symmetric_operator=False, observables=False, periodic=False, transformed_or_original='original', title ='VMC energy of the Haldane chain (N=12) with the InverseSampler')
#plotObservables('results/InverseSampler/FFNN/L12_observables.csv', L=12, hamiltonian='transformed_Heisenberg', observable='StringCorr')
#plot('results/VBSSampler/FFNN/L12.log', L=12, symmetric_operator=False, observables=False, periodic=False, transformed_or_original='original', title ='VMC energy of the Haldane chain (N=12) with the VBSSampler')


#transformed AKLT results
#plot('run/transformedAKLT/DeepConvNN/L40.log', L=40, symmetric_operator=False, observables=False, transformed_or_original='AKLT', title='VMC energy of transformed AKLT model (N=40)')
#plotObservables('run/transformedAKLT/DeepConvNN/L40_observables.csv', L=40, hamiltonian='AKLT', title='String order parameter for the transformed AKLT chain (N=40)', yLabel='String order parameter')
#plot('results/transformedAKLT/DeepConvNN/L60.log', L=60, symmetric_operator=False, observables=False, transformed_or_original='AKLT', title='VMC energy of transformed AKLT model (N=60)')
#plotObservables('results/transformedAKLT/DeepConvNN/L60_observables.csv', L=60, hamiltonian='AKLT', title='String order parameter for the transformed AKLT chain (N=60)', yLabel='String order parameter')


#Scaling of Lanczos Energy
#plotEnergyPerSize()


#Compare number of zeros
#plotS_Z_squared('results/transformedAKLT/DeepConvNN/L60_observables.csv', L=60, title=r'$< \left( S_i^{(z)} \right) ^2 >$ for the AKLT model (N=60)')
#plotS_Z_squared('results/transformedHamiltonian//L60_observables.csv', L=60, title=r'$< \left( S_i^{(z)} \right) ^2 >$ for the Haldane chain (N=60)')

#plotS_Z_squared('results/problemsAKLT/FFNN/L12_observables.csv', L=12)
#plotS_Z_squared('results/problems/FFNN/L12_observables.csv', L=12)


#TransformedFFNN
#plot('results/TransformedFFNN/L4.log', L=4, symmetric_operator=False, observables=False, transformed_or_original='original', title='VMC energy of the Haldane chain (N=4) with the TransformedFFNN')






#compareDMRG('results/transformedHamiltonian/L12_observables.csv', L=12, mode='symmetric')
#compareDMRG('results/transformedHamiltonian/L12_observables.csv', L=12, mode='unsymmetric')
#compareDMRG('results/transformedHamiltonian/L60_observables.csv', L=60, mode='symmetric')
#compareDMRG('results/transformedHamiltonian/L60_observables.csv', L=60, mode='unsymmetric')
#compareDMRG('results/transformedHamiltonian/L80_load_observables.csv', L=80, mode='symmetric')
#compareDMRG('results/transformedHamiltonian/L80_observables.csv', L=80, mode='unsymmetric')


#plotObservables('run/CompareSymmObserv/L20_load8_observables.csv', L=20, observable='FerroCorr')

#compareLatticeSize(L=30)

#for a in [11, 13, 15, 17]:
    #compareNetworkSize(alpha=a)

#compareNetworkSize(alpha=15)

#compareLatticeSizes()

#compareNetworkSizes()


