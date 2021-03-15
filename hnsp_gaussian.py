import numpy as np
import os, time
import scipy

import matplotlib.pyplot as plt
import pandas as pd
import HiercModAna_lib as HMA
import sys

# %%
def gaussian_sim(path, sub_code, out_path):
    # Gaussian simulation
    import numpy as np
    import pandas as pd
    import HiercModAna_lib as HMA
    #  Load data
    print('Load data')
    SCmat = np.genfromtxt(path + '/SC/SC_' + sub_code + '_mat.csv', delimiter='  ')
    waytotal = np.genfromtxt(path + '/SC/WAYTOTAL_' + sub_code + '_mat.csv', delimiter='  ')
    out_path = out_path + 'GAUSSIAN_' + sub_code + '_df.pkl.zip'
    SIMV = pd.DataFrame()
    print('Gaussian simulation')
    N = 360
    g_FCimp = []
    g_Q = []
    # normalize SC
    correct_waytotal = np.repeat(waytotal[:, None], np.shape(waytotal)[0], axis=1)
    SCmat = SCmat / correct_waytotal
    for g in range(-10, 90, 1):  # couplings g
        SCevg = (SCmat + SCmat.conj().T) / 2  # brain structural connectivity matrix
        H = HMA.structural_network(SCevg, N)  # Laplace matrix
        Q, FCipm = HMA.Ideal_Predication_Model(H, g, N)
        FCipm = (FCipm + FCipm.conj().T) / 2  # FC matrix
        g_FCimp += [FCipm]
        g_Q += [Q]
    # store data
    SIMV['Gs_FCimp'] = [g_FCimp]
    SIMV['g_Q'] = [g_Q]
    SIMV.to_pickle(out_path, compression='zip', protocol=4)
    print('Done, saving here: ' + str(out_path))
    return SIMV

# %%
list_sub = pd.read_csv('/user/warm0895/thesis/structural_network_v1/SUB-SES_thesis22-1-21.list', header=None).to_numpy()
path = '/gss/work/warm0895/thesis_dhcpdata/matrices'
out_path_all = '/gss/work/warm0895/thesis_dhcpdata/run_hnsp/allsubs_gaussian_simulation_dataframe.pkl.zip'
out_path = '/gss/work/warm0895/thesis_dhcpdata/run_hnsp/gauss_sim'

for sub in list_sub:
    sub_code = sub[0]+'_'+str(sub[1])
    print(sub_code)
    gaussian_sim(path, sub_code, out_path)


