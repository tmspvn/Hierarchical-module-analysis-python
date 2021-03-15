import numpy as np
import scipy
import pandas as pd
import HiercModAna_lib as HMA


# %%
def HierachicalAnalysis_static(path, sub_code, N=360):  # all subjects togheter
    import numpy as np
    import HiercModAna_lib as HMA
    BOLD = np.genfromtxt(path + '/ROIts_' + sub_code + '_mat.csv', delimiter=',')
    FC = np.corrcoef(BOLD)
    #  static analysis
    print('static analysis')
    Clus_num, Clus_size, mFC = HMA.functional_hp(FC, N)
    Hin_static, Hse_static, p = HMA.balance(FC, N, Clus_num, Clus_size)
    return [sub_code, Clus_num, Clus_size, mFC, Hin_static, Hse_static, p]

# %% part 1
list_sub = pd.read_csv('/user/warm0895/thesis/structural_network_v1/SUB-SES_thesis22-1-21.list', header=None).to_numpy()
path = '/gss/work/warm0895/thesis_dhcpdata/matrices/FC'
out_path = '/gss/work/warm0895/thesis_dhcpdata/run_hnsp/allsubs_single_static_analysis_nocalibration_dataframe.pkl.zip'
# %%
static_results = []
long_fmri = []
for sub in list_sub:
    sub_code = sub[0]+'_'+str(sub[1])
    print(sub_code)
    static_results += [HierachicalAnalysis_static(path, sub_code, N=360)]
    BOLD = np.genfromtxt(path + '/ROIts_' + sub_code + '_mat.csv', delimiter=',')
    long_fmri += [BOLD]


static_results = pd.DataFrame(static_results,
                              columns=['sub_code', 'Clus_num', 'Clus_size', 'mFC', 'Hin_static', 'Hse_static', 'p'])
static_results = static_results.reset_index().set_index('sub_code')
static_results.to_pickle(out_path, compression='zip', protocol=4)


#compute long fmri
N = 360
long_fmri = np.concatenate(long_fmri, axis=1)
longFC = np.corrcoef(long_fmri)
np.savetxt("/gss/work/warm0895/thesis_dhcpdata/run_hnsp/superlongfmri_for_calibration_FC.csv", longFC, delimiter=",")