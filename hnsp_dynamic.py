import numpy as np
import scipy
import pandas as pd
import HiercModAna_lib as HMA
import sys

sub_code = sys.argv[1]
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


def HierachicalAnalysis_dynamic(path, sub_code, allsub_static_analysis, out_path, N=360, window_size=154):  # subject wise
    import numpy as np
    import pandas as pd
    import HiercModAna_lib as HMA
    #  Load data
    print('Load data')
    BOLD = np.genfromtxt(path + '/FC/ROIts_' + sub_code + '_mat.csv', delimiter=',')
    SCmat = np.genfromtxt(path + '/SC/SC_' + sub_code + '_mat.csv', delimiter='  ')
    waytotal = np.genfromtxt(path + '/SC/WAYTOTAL_' + sub_code + '_mat.csv', delimiter='  ')
    FC = np.corrcoef(BOLD)
    allsub_static_analysis = pd.read_pickle(allsub_static_analysis)
    longFC = np.genfromtxt('/gss/work/warm0895/thesis_dhcpdata/run_hnsp/superlongfmri_for_calibration_FC.csv', delimiter=',')

    out_path = out_path + '/RESULTS_' + sub_code + '_df.pkl.zip'
    SUB_RESULTS = pd.DataFrame()
    SUB_RESULTS['sub_code'] = [sub_code]

    # load all sub static results
    IN_static = allsub_static_analysis['Hin_static']
    IM_static = allsub_static_analysis['Hse_static']
    # Calibration
    Clus_num, Clus_size, mFC, Hin_static, Hse_static, p = HMA.stable_correct(sub_code, longFC, IN_static, IM_static, N)
    # store data
    for z in zip(['Clus-num', 'Clus_size', 'mFC', 'Hin_static', 'Hse_static', 'p'],
                 [Clus_num, Clus_size, mFC, Hin_static, Hse_static, p]):
        SUB_RESULTS[z[0]] = [z[1]]

    #  dynamic analysis -> 2300vol @ 15 min -> (15*60)/2300=0.39130 == TR
    print('dynamic analysis part 1')
    Total = 2300
    width = window_size  # 154 # -> 1 minute
    IN = []
    IM = []
    for t in range(0, Total - width):
        subdata = BOLD[:, t:t + width]  # moving window
        FC = np.corrcoef(subdata)
        Clus_num, Clus_size, FCn = HMA.functional_hp(FC, N)
        Hin, Hse, p = HMA.balance(FC, N, Clus_num, Clus_size)
        IN += [Hin]
        IM += [Hse]
        print(Total - width - t)
    # store data
    for z in zip(['IN_nocalb', 'IM_nocalib'], [IN, IM]):
        SUB_RESULTS[z[0]] = [z[1]]

    #  dynamic analisys part 2
    print('dynamic analisys part 2')
    TR = 0.39130
    Hin = HMA.individual_correction(np.array(IN), Hin_static)  # Hin CALIBRATED from static for this subj
    Hse = HMA.individual_correction(np.array(IM), Hse_static)  # Hse CALIBRATED from static for this subj
    Fre, DIn, DSe, In_time, Se_time = HMA.Flexible(Hin - Hse, TR)  # calculating dynamic measures, @2300 timepoints
    Z = [Fre, DIn, DSe, In_time, Se_time]
    # store data
    SUB_RESULTS['Z: Fre,DIn,DSe,In_time,Se_time'] = [Z]
    for z in zip(['Hin_dynamic_calibrated', 'Hse_dynamic_calibrated'], [Hin, Hse]): # save Hin and
        SUB_RESULTS[z[0]] = [z[1]]

    # Gaussian simulation
    print('Gaussian simulation')
    N = 360
    g_FCimp = []
    g_Q = []
    # normalize SC
    correct_waytotal = np.repeat(waytotal[:, None], np.shape(waytotal)[0], axis=1)
    SCmat = SCmat / correct_waytotal
    for g in range(10, 150, 5):  # couplings g
        SCevg = (SCmat + SCmat.conj().T) / 2  # brain structural connectivity matrix
        H = HMA.structural_network(SCevg, N)  # Laplace matrix
        Q, FCipm = HMA.Ideal_Predication_Model(H, g, N)
        FCipm = (FCipm + FCipm.conj().T) / 2  # FC matrix
        g_FCimp += [FCipm]
        g_Q += [Q]
    # store data
    SUB_RESULTS['Gs_FCimp'] = [g_FCimp]
    SUB_RESULTS['g_Q'] = [g_Q]
    SUB_RESULTS.to_pickle(out_path, compression='zip', protocol=4)
    print('Done, saving here: ' + str(out_path))
    return SUB_RESULTS


# %% part 1

print(sub_code)
path = '/gss/work/warm0895/thesis_dhcpdata/matrices/'
allsub_static_analysis = '/gss/work/warm0895/thesis_dhcpdata/run_hnsp/allsubs_single_static_analysis_nocalibration_dataframe.pkl.zip'
out_path = '/gss/work/warm0895/thesis_dhcpdata/run_hnsp/singlesub_results'

HierachicalAnalysis_dynamic(path, sub_code, allsub_static_analysis, out_path, N=360, window_size=154)
