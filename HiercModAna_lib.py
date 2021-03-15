#import numpy as np
#import os
#import scipy
#import matplotlib.pyplot as plt
#import pandas as pd

# python implementation of functions needed to run hierarchical modular partitioning analysis
# source: https://github.com/TobousRong/Hierarchical-module-analysis
# v 1, 11 feb 2021, Oldenburg

#
#ROIts = np.genfromtxt('ROIts_CC00907XX16_4230_mat.csv', delimiter=',')
#SCmat = np.genfromtxt('SC_CC00907XX16_4230_mat.csv', delimiter='  ')
#matlab_eigv = np.genfromtxt('./HiModAn_py/ev.csv', delimiter=',')
#FC = np.corrcoef(ROIts)
#N = 360
#%% FUNCTIONAL_HP.M -> tested
def functional_hp(FC, N=360):
    import numpy as np
    import pandas as pd
    # This function performs the hierarchical module partition of FC networks using the NSP method.
    # Input:
    # data--original FC matrix; N-- number of ROI
    # Output:
    # Clus_num---modular number in each level, normalized to [0 1];
    # Clus_size-- the size (e.g., ROI number) of each module in each level;
    # FC-- the optimized FC network after the hierarchical modular partition, the regions are reordered.
    # inputs :
    data, N = FC, N
    # This method requires the complete positive connectivity in FC matrix,
    # that generates the global integration in the first level.
    data[data < 0] = 0
    data = (data+data.conj().T)/2
    FE, FEC = np.linalg.eig(data) # eigenvalues and eigenvector out from numpy are reversed to matlab
    # sort descending? no need they are ordered correctly in python
    #FEC = np.flip(FEC)
    H0_0 = np.where(FEC[:, 0] < 0)[0]
    H0_1 = np.where(FEC[:, 0] >= 0)[0]
    # The first level has one module and corresponds to the global integration.
    Clus_num = [1] # cluster number 1
    Clus_size = []
    # is FE diag mat?
    #empty_mat = np.zeros_like(data)
    #np.fill_diagonal(empty_mat, FE)
    #FE = np.flip(empty_mat)
    #FEC = matlab_eigv
    for mode in range(1, N+1): # bisection "counter",
        x = np.where(FEC[:, mode] >= 0)[0]
        y = np.where(FEC[:, mode] < 0)[0]
        H = pd.DataFrame()
        for j in range(0, 2*Clus_num[mode-1]):
            # assume the number of cluster in j-1 level is 2^(mode-1)
            # so each level j-1 has the number of clusters in mode-1 times 2
            H['H' + str(mode-1) + '_' + str(j)] = [np.array(eval('H' + str(mode-1) + '_'+ str(j)))]

        id = [len(val) for val in H[H.columns].to_numpy()[0]]  # length of each cluster in H
        keep0s = np.where(np.array(id) != 0)[0] # keep only non zero cluisters
        Clus_size += [np.array(id)[keep0s]] # list of size of clusters
        rm0s = np.where(np.array(id) == 0) # empty clusters
        H = H.drop(columns=H.columns[rm0s]) # drop empty clusters
        Clus_num += [ H.size ]  # number of cluster ina level

        for k, j in enumerate(range(0, 2 * Clus_num[mode], 2)): # Intersect currect cluster(mode) with previous cluster(mode-1) brought by H
            Positive_Node = np.intersect1d(x, H[H.columns[k]].to_numpy()[0])
            Negative_Node = np.intersect1d(y, H[H.columns[k]].to_numpy()[0])
            exec('H' + str(mode) + '_' + str(j + 1) + ' = ' + 'Positive_Node')
            exec('H' + str(mode) + '_' + str(j) + ' = ' + 'Negative_Node')

        for j in range(0, 2 * Clus_num[mode - 1]): #delete variables
            exec('del' + ' H' + str(mode - 1) + '_' + str(j))
        Z = []
        if (Clus_num[-1] == N): # save structure of the cluster in the last level
            for j in range(0, 2 * Clus_num[mode]):
                Z += [eval('H' + str(mode) + '_' + str(j))]
            # Delete last variables
            for j in range(0, 2 * Clus_num[mode]):
                exec('del' + ' H' + str(mode) + '_' + str(j))
            break

    Clus_num= Clus_num[1:]
    Clus_num = np.array(Clus_num)/N
    c = np.ones([N])
    c[0:len(Clus_num)] = Clus_num
    Clus_num = c
    FCn = np.zeros([N,N])
    z = [elm for elm in Z if elm.shape[0] != 0]
    z = np.stack(z).squeeze()
    # Reorder the matrix
    for i in range(0, N):
        for j in range(0, N):
            FCn[i, j] = data[z[i], z[j]]
    return Clus_num, Clus_size, FCn

# %% BALANCE.M -> tested
def balance(FC, N, Clus_num, Clus_size):
    import numpy as np
    import pandas as pd
    #function [Hin,Hse,p] =Balance(FC,N,Clus_size,Clus_num)
    # This function calculates the integration and segregation component
    # input: FC-- functional matrix, N-- number of ROI
    # Clus_size-- modular size in each level, Clus_num-- modular number;
    # Clus_size and Clus_num are calcuated from the functuon 'Functional_HP'
    # output: Hin--integration component, Hse-- segregation component, p-- correction fator of modular size

    FC = (FC + FC.conj().T) / 2
    FC[FC < 0] = 0
    [FE, FEC] = np.linalg.eig(FC) # FE/1000 = matlab FE!!?? doens't change the result
    FE[FE < 0] = 0
    FE = FE ** 2 # using the squared Lambda, ask question
    p = np.zeros([N])
    for i in range(0, len(np.where(Clus_num<1)[0])):          #i=1:length(find(Clus_num<1))
          p[i] = np.sum(np.abs(Clus_size[i]-1 / Clus_num[i] )) / N # modular size correction

    HF = FE[np.newaxis,:] * Clus_num[np.newaxis,:] * (1-p[np.newaxis,:]) # element-wise
    HF = HF.squeeze()
    Hin = np.sum(HF[0]) / N  # integration component
    Hse = np.sum(HF[1:N]) / N  # segregation component
    return Hin, Hse, p

# %% STABLE_CORRECT.M
# Calibrating the individual static segregation and integration component
# compute mean Hin and Hse of the multiple subjects. IN and IM are Hse_static and Hin_static of multiple subjects
# So thge correction is based on multiple babies
# Is it necessary for me? YES
def stable_correct(sub_code, sFC, IN, IM, N):
    import numpy as np
    import HiercModAna_lib as HMA
    # input: sFC-- stable FC matrix from long-enough fMRI time; IN-- individual static integration component;
    # IM-- individual static segregation component; N-- number of ROI
    # output: Hin-- calibrated individual integration component; Hse-- calibrated individual segregation component
    Clus_num, Clus_size, mFC = HMA.functional_hp(sFC, N)
    R_IN, R_IM, p = HMA.balance(sFC, N, Clus_num, Clus_size)
    #integration component R_IN and segregation component R_IM for stable Fc matrix.
    # Proportional calibration scheme. Since our Gaussian model has proved a theoretical functional balance,
    # the mean individual integration and segregation components were calibrated to the equal value, R_IN.
    p2 = (np.mean(IM) - R_IN) / np.mean(IM)
    Hse = IM[IN.index == sub_code] * (1-p2)
    p1 = (np.mean(IN) - R_IN) / np.mean(IN)
    Hin = IN[IN.index == sub_code] * (1 - p1)
    return Clus_num, Clus_size, mFC, Hin[0], Hse[0], [p1, p2]

# %% STRUCTURAL NETWORK.M -> tested
def structural_network(A, N=360):
    import numpy as np
    # Create laplacian matrix for structural connectivity
    # remember to (SCmat + SCmat.conj().T)/2 before inputting A -> A=(SCmat + SCmat.conj().T)/2
    A = -A
    for i in range(0, N):
        A[i, i] = -np.sum(A[i, :])
    B = A / np.max(np.linalg.eig(A)[0]) # np.eig(A)[0] -> eig vals, np.eig(A)[1] -> R eig vects
    return B

# %% Ideal_Predication_Model.M -> tested
# inputs: B = laplace mat of structural connectivity
#         g = g parameter to be interated outside
#         N = number areas (360)
def Ideal_Predication_Model(B, g, N=360):
    import numpy as np
    Q = np.linalg.matrix_power((np.eye(N) + g * B), -1)  # matlab ^ is matrix_power, while .^ = **, elementwise
    Q = Q @ Q.conj().T
    C = np.zeros([N, N])
    for i in range(0, N):
        for j in range(0, N):
            C[i, j] = Q[i, j] / np.sqrt(Q[i, i] * Q[j, j])
    return Q, C

# %% individual_correction.m
def individual_correction(IN,R_IN):
    import numpy as np
# This function calibrates the dynamic Hin and Hse of each individual to their static values that have been calibrated.
# input: IN-- dynamic integration component (or segregation component) for a subject;
# R_IN-- the calibrated static integration component (or segregation component) for the subject;
# output: calibrated integration component (or segregation component) for the subject;
    p = (np.mean(IN) - R_IN) / np.mean(IN)
    Hin = IN * (1-p)
    return Hin

# %% Flexible.m
# compute dynamic measures
# HB = Hin-Hse,  TR = 0.72 in the paper, is it fMRI sampling frequency. -> TR=0.39130
def Flexible(HB,TR=0.39130, window_size=154):
    import numpy as np
    #   strength: amplitude of dynamic deviation from the balanced state to the integrated state or segregated state
    DIn = np.sum(HB[np.where(HB > 0)])
    DSe = np.sum(HB[np.where(HB < 0)])
    #   dwell time in segregated and integrated states
    In_time = len(np.where(HB >= 0)[0]) / (2300-window_size)  # -> 2300vol @ 15 min -> (15*60)/2300=0.39130
    Se_time = len(np.where(HB < 0 )[0]) / (2300-window_size)
    #   transition frequency / switching frequency. transition speed between segregated and integrated states
    HB[HB < 0] = -1
    HB[HB > 0] = 1
    Fre = len(np.where(np.abs(np.diff(HB)) > 0)[0]) / ((2300-window_size) * TR) # TR transforms timepoints to seconds
    return Fre, DIn, DSe, In_time, Se_time


#%% test, static analysis
#Clus_num, Clus_size, FCn = functional_hp(FC, N)
#Hin, Hse = balance(FC, N, Clus_num, Clus_size)

#%% test, gaussian simulation
#SC = (SCmat + SCmat.conj().T)/2
#B = structural_network(SC)
#Q, C = Ideal_Predication_Model(B, 10)

# %%







