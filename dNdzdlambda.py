import numpy as np
from multiprocessing import Pool
from scipy.stats import norm
from scipy.interpolate import InterpolatedUnivariateSpline
import scaling_relations


# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return DistCompute.run_field(*arg)


################################################################################
class DistCompute:

    def __init__(self, SPT_survey,
                 z_cl_min_max,
                 lambda_min,
                 NPROC):
        self.SPT_survey = SPT_survey
        self.z_cl_min_max = z_cl_min_max
        self.lambda_min = lambda_min
        self.NPROC = NPROC
        # Observable arrays for output
        self.lambda_bins_out = np.logspace(np.log10(8.4), np.log10(250), 16)

    def run(self, HMF, cosmology, scaling):
        """Return ln-likelihood for SPT cluster abundance."""
        self.HMF = HMF
        self.cosmology = cosmology
        self.scaling = scaling
        # observables[z,M]
        self.lnrichness_m = scaling_relations.lnmass2lnobs('richness', self.HMF['lnM_arr'][None, :],
                                                           self.HMF['z_arr'][:, None], self.scaling)
        # Compute distribution for each SPT field (optional multiprocessing)
        num_fields = len(self.SPT_survey)
        if self.NPROC == 0:
            field_results = [self.run_field(fieldidx) for fieldidx in range(num_fields)]
        else:
            with Pool(processes=self.NPROC) as pool:
                argin = zip([self]*num_fields, range(num_fields))
                field_results = pool.map(unwrap_self_f, argin)
        N_lambda = np.array([field_results[i] for i in range(num_fields)]).sum(axis=0)

        return N_lambda

    ##########

    def run_field(self, fieldidx):
        """Return dN/dz and dN/dlambda for a given SPT field (index)."""
        # Survey field
        if 'noMCMF' in self.SPT_survey['FIELD'][fieldidx]:
            return np.zeros(len(self.lambda_bins_out)-1)
        lnzeta_m = scaling_relations.lnmass2lnobs('zeta', self.HMF['lnM_arr'][None, :], self.HMF['z_arr'][:, None],
                                                  self.scaling, self.cosmology, SPTfield=self.SPT_survey[fieldidx])
        # dN/dlnlambda/dlnzeta
        dN_dz_dlnobs = (np.exp(self.HMF['richness_SZ_dNdlnM'])
                        * scaling_relations.dlnM_dlnobs('zeta', self.scaling)
                        * scaling_relations.dlnM_dlnobs('richness', self.scaling, z=self.HMF['z_arr'])
                        * (np.pi/180)**2 * self.SPT_survey['AREA'][fieldidx])
        # xi|M
        this_xi_m = scaling_relations.zeta2xi(np.exp(lnzeta_m))
        # Cut in zeta
        dN_dz_dlnobs[lnzeta_m[:, None, :]*np.ones(dN_dz_dlnobs.shape) < np.log(self.scaling['zeta_min'])] = 0
        # P(xi>cut)
        dN_dz_dlnobs *= norm.cdf(this_xi_m, self.SPT_survey['XI_MIN'][fieldidx], 1)[:, None, :]
        # Integrate out zeta [z,lambda]
        dN_dz_dlnrichness = np.sum(.5 * (dN_dz_dlnobs[:, :, 1:]+dN_dz_dlnobs[:, :, :-1])
                                   * (lnzeta_m[:, 1:]-lnzeta_m[:, :-1])[:, None, :], axis=2)
        # Cut in lambda
        lambda_min_allz = self.lambda_min[self.SPT_survey['LAMBDA_MIN'][fieldidx]](self.HMF['z_arr'])
        # N(Delta lnlambda)
        N_z_lambda = np.zeros((len(self.HMF['z_arr']), len(self.lambda_bins_out)-1))
        for i in range(len(self.HMF['z_arr'])):
            f = InterpolatedUnivariateSpline(self.lnrichness_m[i, :], dN_dz_dlnrichness[i, :])
            N_z_lambda[i] = np.array([f.integral(np.log(self.lambda_bins_out[j]), np.log(self.lambda_bins_out[j+1]))
                                      for j in range(len(self.lambda_bins_out)-1)])
            lambda_min_idx = np.digitize(lambda_min_allz[i], self.lambda_bins_out)-1
            N_z_lambda[i, :lambda_min_idx] = 0.
            N_z_lambda[i, lambda_min_idx] = f.integral(np.log(lambda_min_allz[i]),
                                                       np.log(self.lambda_bins_out[lambda_min_idx+1]))
        N_lambda = np.array([InterpolatedUnivariateSpline(self.HMF['z_arr'], N_z_lambda[:, i]
                                                          ).integral(self.z_cl_min_max[0], self.z_cl_min_max[1])
                             for i in range(len(self.lambda_bins_out)-1)])
        return N_lambda
