from __future__ import division
import numpy as np
from cosmosis.datablock import option_section
import compute_HMF
import pickle

def setup(options):
    recalc_HMF = options.get_bool(option_section, 'recalc_HMF', default=T)
    save_HMF_to_disk = options.get_bool(option_section, 'save_HMF_to_disk', default=F)
    if recalc_HMF:
        Deltacrit = options.get_double(option_section, 'Deltacrit', default=500.)
        HMF_calculator = compute_HMF.HMFCalculator(Deltacrit)
    else:
        HMF_calculator = pickle.load(open('HMF.pkl', 'rb'))
    return HMF_calculator

def execute(block, HMF_calculator):
    if recalc_HMF:
        HMF_calculator.compute_HMF(block)
        if save_HMF_to_disk:
            HMF = {'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
                'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
                'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
            pickle.dump(HMF, open('HMF.pkl', 'wb'))
    else:
        block.put_double_array_1d('HMF', 'M_arr', HMF_calculator['M_arr'])
        block.put_double_array_1d('HMF', 'z_arr', HMF_calculator['z_arr'])
        block.put_double_array_nd('HMF', 'dNdlnM', HMF_calculator['dNdlnM'])
    return 0

def cleanup(config):
    pass
