def setup(options):
    pass


def execute(block):
    theta = block.get_double('cosmological_parameters', 'cosmomc_theta')
    lnL = -.5 * ((theta-1.04109)/0.00030)**2
    block.put_double('likelihoods', 'THETA_LIKE', lnL)
    return 0


def cleanup(config):
    pass
