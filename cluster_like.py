import numpy as np


ln2pi = np.log(2.*np.pi)


class ClusterLike:
    def __init__(self, z_bins, SNR_bins, catalog):
        """Bin the cluster `catalog`."""
        SNR = np.sqrt(catalog['XI']**2-3)/catalog['GAMMA_FIELD']
        self.N_data = np.histogram2d(catalog['REDSHIFT'], SNR,
                                     bins=(z_bins, SNR_bins))[0].flatten()
        self.dim = len(self.N_data)

    def lnlike(self, N_model, cov_samplevar):
        """Return the ln-likelihood."""
        cov = cov_samplevar + np.diag(N_model)
        diff = N_model - self.N_data
        lnlike = -.5 * (np.dot(diff, np.linalg.solve(cov, diff))
                        + self.dim*ln2pi
                        + np.linalg.slogdet(cov)[1])
        return lnlike
