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
                 z_DESWISE,
                 NPROC):
        self.SPT_survey = SPT_survey
        self.surveyCutRedshift = surveyCutRedshift
        self.surveyCutRichness = surveyCutRichness
        self.z_DESWISE = z_DESWISE,
        self.NPROC = NPROC

        ##### Observable arrays for output
        self.lnlambda_bins = np.linspace(np.log(9), np.log(250), 151)
        self.Nlambda = (len(self.lnlambda_bins)//10)+1
        dz = .1
        self.Nz = int((self.surveyCutRedshift[1]-self.surveyCutRedshift[0])/dz + 1)
        self.z_bins_output = np.linspace(self.surveyCutRedshift[0], self.surveyCutRedshift[1], self.Nz)

    def run(self, HMF, cosmology, scaling):
        """Return ln-likelihood for SPT cluster abundance."""
        self.HMF = HMF
        self.cosmology = cosmology
        self.scaling = scaling
        # observables[z,M]
        self.lnzeta_m = {'500d': scaling_relations.lnmass2lnobs('zeta', self.HMF['lnM_arr'][None,:], self.HMF['z_arr'][:,None], self.scaling, self.cosmology, SPTsurvey='500d'),
                         'ECS': scaling_relations.lnmass2lnobs('zeta', self.HMF['lnM_arr'][None,:], self.HMF['z_arr'][:,None], self.scaling, self.cosmology, SPTsurvey='ECS'),
                         'SZ': scaling_relations.lnmass2lnobs('zeta', self.HMF['lnM_arr'][None,:], self.HMF['z_arr'][:,None], self.scaling, self.cosmology, SPTsurvey='SZ'),}
        self.lnrichness_m = scaling_relations.lnmass2lnobs('richness_base', self.HMF['lnM_arr'][None,:], self.HMF['z_arr'][:,None], self.scaling)
        lnrichness_m_ext = scaling_relations.lnmass2lnobs('richness_ext', self.HMF['lnM_arr'][None,:], self.HMF['z_arr'][:,None], self.scaling)
        self.lnrichness_m[self.HMF['z_arr']>self.z_DESWISE,:] = lnrichness_m_ext[self.HMF['z_arr']>self.z_DESWISE,:]

        ##### Compute distribution for each SPT field (optional multiprocessing)
        num_fields = len(self.SPT_survey)
        if self.NPROC==0:
            field_results = [self.run_field(fieldidx) for fieldidx in range(num_fields)]
        else:
            with Pool(processes=self.NPROC) as pool:
                argin = zip([self]*num_fields, range(num_fields))
                field_results = pool.map(unwrap_self_f, argin)
        dN_dz = np.array([field_results[i][0] for i in range(num_fields)]).sum(axis=0)
        dN_dlnlambda = np.array([field_results[i][1] for i in range(num_fields)]).sum(axis=0)

        return dN_dz, dN_dlnlambda

    ##########

    def run_field(self, fieldidx):
        """Return dN/dz and dN/dlambda for a given SPT field (index)."""
        # Survey field
        if 'noMCMF' in self.SPT_survey['FIELD'][fieldidx]:
            return np.zeros(self.Nz), np.zeros(self.Nlambda)
        elif self.SPT_survey['FIELD'][fieldidx]=='sptpol_500d_MCMF':
            this_lnzeta_m = self.lnzeta_m['500d'] + np.log(self.SPT_survey['GAMMA'][fieldidx])
        else:
            if '_sptpol' in self.SPT_survey['FIELD'][fieldidx]:
                this_lnzeta_m = self.lnzeta_m['ECS'] + np.log(self.SPT_survey['GAMMA'][fieldidx]) + np.log(self.scaling['SPECS_calib'])
            else:
                this_lnzeta_m = self.lnzeta_m['SZ'] + np.log(self.SPT_survey['GAMMA'][fieldidx])
        # dN/dlnlambda/dlnzeta
        dN_dz_dlnobs = np.exp(self.HMF['richness_SZ_dNdlnM']) * scaling_relations.dlnM_dlnobs('zeta', self.scaling) * (np.pi/180)**2 * self.SPT_survey['AREA'][fieldidx]
        dN_dz_dlnobs[self.HMF['z_arr']<self.z_DESWISE,:,:]*= scaling_relations.dlnM_dlnobs('richness_base', self.scaling)
        dN_dz_dlnobs[self.HMF['z_arr']>=self.z_DESWISE,:,:]*= scaling_relations.dlnM_dlnobs('richness_ext', self.scaling)
        # xi|M
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
        # dN_dz = np.sum(np.exp(.5*(lndN_dz_dlnrichness[:,1:]+lndN_dz_dlnrichness[:,:-1])) * np.diff(self.lnrichness_m), axis=1)
        dN_dz_out = np.zeros(self.Nz)# interp1d(self.HMF['z_arr'], dN_dz, kind='linear')(self.z_bins_output)
        dN_dlnlambda = (np.trapz(np.exp(lndN_dz_dlnrichness_grid), self.HMF['z_arr'], axis=0))[::10]
        return dN_dz_out, dN_dlnlambda
