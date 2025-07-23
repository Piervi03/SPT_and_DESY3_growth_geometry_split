import numpy as np

from cosmosis.datablock import option_section

import meanlnmass_like

def setup(options):
    z_bins = options.get_double_array_1d(option_section, 'SPTcl_z_bins')
    SNR_bins = options.get_double_array_1d(option_section, 'SPTcl_SNR_bins')
    data_file = options.get_string(option_section, 'meanlnmass_file')
    tmp = np.loadtxt(data_file)
    mean, cov = tmp[0], tmp[1:]
    return mean, cov


def execute(block, config):
    model = block.get_double_array_1d('mean_lnmass', 'lnM')
    lnlike = meanlnmass_like.lnlike(mean, model, cov)
    block.put_double('likelihoods', 'MEANLNMASS_LIKE', lnlike)
    return 0


def cleanup(config):
    pass
