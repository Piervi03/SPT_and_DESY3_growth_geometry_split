import numpy as np
from astropy.table import Table

from cosmosis.datablock import option_section

import cluster_like

def setup(options):
    catalog = Table.read(options.get_string(option_section, 'SPTcatalogfile'))
    catalog = catalog[catalog['COSMO_SAMPLE'] == 1]
    z_bins = options.get_double_array_1d(option_section, 'SPTcl_z_bins')
    SNR_bins = options.get_double_array_1d(option_section, 'SPTcl_SNR_bins')
    cluster_like_func = cluster_like.ClusterLike(z_bins, SNR_bins, catalog)
    cov_samplevar_file = options.get_string(option_section, 'cov_samplevar')
    cov_samplevar = np.loadtxt(cov_samplevar_file)
    return {'cluster_like_func': cluster_like_func,
            'cov_samplevar': cov_samplevar}


def execute(block, config):
    N_model = block.get_double_array_1d('SPT_cluster', 'N')
    lnlike = config['cluster_like_func'].lnlike(N_model, config['cov_samplevar'])
    block.put_double('likelihoods', 'CLUSTER_LIKE', lnlike)
    return 0


def cleanup(config):
    pass
