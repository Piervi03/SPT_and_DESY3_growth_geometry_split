from __future__ import division
import numpy as np
from multiprocessing import Pool
from scipy.stats import norm
from scipy.interpolate import interp1d
import scaling_relations


# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return DistCompute.run_field(*arg)


################################################################################
class DistCompute:

    def __init__(self, SPT_survey,
                 surveyCutRedshift,
                 surveyCutRichness,
                 NPROC):
        self.SPT_survey = SPT_survey
        self.surveyCutRedshift = surveyCutRedshift
        self.surveyCutRichness = surveyCutRichness
        self.NPROC = NPROC

        ##### Observable arrays for output
        self.lnlambda_bins = np.linspace(np.log(10), np.log(250), 151)
        self.Nlambda = (len(self.lnlambda_bins)//10)+1
        dz = .1
        self.Nz = int((self.surveyCutRedshift[1]-self.surveyCutRedshift[0])/dz + 1)
        self.z_bins_output = np.linspace(self.surveyCutRedshift[0], self.surveyCutRedshift[1], self.Nz)

    def run(self, HMF, cosmology, scaling):
        """Return ln-likelihood for SPT cluster abundance."""
        self.HMF = HMF
        self.cosmology = cosmology
        self.scaling = scaling
        # HMF in obs space [z,richness,SZ]
        self.dN_dlnobs_deg2 = np.exp(self.HMF['dNdlnM']) * scaling_relations.dlnM_dlnobs('richness', self.scaling)*scaling_relations.dlnM_dlnobs('zeta', self.scaling) * (np.pi/180)**2
        # observables[z,M]
        self.lnzeta_m = scaling_relations.lnmass2lnobs('zeta', self.HMF['lnM_arr'][None,:], self.HMF['z_arr'][:,None], self.scaling, self.cosmology)
        self.lnrichness_m = scaling_relations.lnmass2lnobs('richness', self.HMF['lnM_arr'][None,:], self.HMF['z_arr'][:,None], self.scaling)

        ##### Compute distribution for each SPT field (optional multiprocessing)
        num_fields = len(self.SPT_survey)
        if self.NPROC==0:
            field_results = [self.run_field(fieldidx) for fieldidx in range(num_fields)]
        else:
            with Pool(processes=self.NPROC) as pool:
                argin = zip([self]*num_fields, range(num_fields))
                field_results = pool.map(unwrap_self_f, argin)
        dN_dz = np.array([field_results[i][0] for i in range(num_fields)]).sum(axis=0)
        dN_dlambda = np.array([field_results[i][1] for i in range(num_fields)]).sum(axis=0)

        return dN_dz, dN_dlambda

    ##########

    def run_field(self, fieldidx):
        """Return dN/dz and dN/dlambda for a given SPT field (index)."""
        if 'noMCMF' in self.SPT_survey['FIELD'][fieldidx]:
            return np.zeros(self.Nz), np.zeros(self.Nlambda)
        # Field size
        dN_dz_dlnobs = self.dN_dlnobs_deg2 * self.SPT_survey['AREA'][fieldidx]
        # xi|M
        this_lnzeta_m = self.lnzeta_m + np.log(self.SPT_survey['GAMMA'][fieldidx])
        if '_sptpol' in self.SPT_survey['FIELD'][fieldidx]:
            this_lnzeta_m+= np.log(self.scaling['SPECS_calib'])
        this_xi_m = scaling_relations.zeta2xi(np.exp(this_lnzeta_m))
        # Cut in zeta
        dN_dz_dlnobs[(this_lnzeta_m[:,None,:]*np.ones(dN_dz_dlnobs.shape)<np.log(self.scaling['zeta_min']))] = 0
        # P(xi>cut)
        dN_dz_dlnobs*= norm.cdf(this_xi_m, self.SPT_survey['XI_MIN'][fieldidx], 1)[:,None,:]
        # Integrate out zeta [z,lambda]
        dN_dz_dlnrichness = np.sum(.5*(dN_dz_dlnobs[:,:,1:]+dN_dz_dlnobs[:,:,:-1])*(this_lnzeta_m[:,1:]-this_lnzeta_m[:,:-1])[:,None,:], axis=2)
        # Cut in lambda
        if self.SPT_survey['FIELD'][fieldidx]=='sptpol_500d_MCMF':
            lambda_min = self.surveyCutRichness['deep'](self.HMF['z_arr'])
        else:
            lambda_min = self.surveyCutRichness['shallow'](self.HMF['z_arr'])
        dN_dz_dlnrichness[self.lnrichness_m<np.log(lambda_min)[:,None]] = 0.
        with np.errstate(all='ignore'):
            lndN_dz_dlnrichness = np.log(dN_dz_dlnrichness)
        # dN/dlambda go to fixed grid in z,lambda and integrate over z; return sparse array
        with np.errstate(divide='ignore'):
            lndN_dz_dlnrichness_grid = np.array([np.interp(self.lnlambda_bins, self.lnrichness_m[i], lndN_dz_dlnrichness[i]) for i in range(len(self.HMF['z_arr']))])
        # dN/dz and dN/dlambda
        dN_dz = np.sum(np.exp(.5*(lndN_dz_dlnrichness[:,1:]+lndN_dz_dlnrichness[:,:-1])) * np.diff(self.lnrichness_m), axis=1)
        dN_dz_out = interp1d(self.HMF['z_arr'], dN_dz, kind='linear')(self.z_bins_output)
        dN_dlambda = (np.trapz(np.exp(lndN_dz_dlnrichness_grid), self.HMF['z_arr'], axis=0)/np.exp(self.lnlambda_bins))[::10]
        return dN_dz_out, dN_dlambda
