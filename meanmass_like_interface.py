import numpy as np

from cosmosis.datablock import option_section

import meanmass_like


def setup(options):
    data_file = options.get_string(option_section, 'meanmass_file')
    tmp = np.loadtxt(data_file)
    return {'mean': tmp[0], 'cov': tmp[1:]}


def execute(block, config):
    model = block.get_double_array_1d('mean_mass', 'M')
    lnlike = meanmass_like.lnlike(config['mean'], model, config['cov'])
    block.put_double('likelihoods', 'MEANMASS_LIKE', lnlike)
    return 0


def cleanup(config):
    pass
