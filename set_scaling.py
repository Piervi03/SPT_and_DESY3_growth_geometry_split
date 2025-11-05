import numpy as np
from math import sqrt as msqrt
from astropy.table import Table

import scaling_relations


class SetScaling:

    def __init__(self, Megacamcalib, HSTcalibfile):
        self.WLcalib = Megacamcalib
        self.HSTcalib = Table.read(HSTcalibfile, format='ascii.commented_header')

    def execute(self, todo, scaling):
        """Set total (or effective) bias and scatter for Megacam and DES using
        the simulation calibration numbers and the nuissance parameters. Set
        possible covariance matrices between all observables we're currently
        analyzing. The scatter in velocity dispersions depends on cluster
        properties and therefore cannot be pre-computed. Return: (bool) whether
        or not all covariance matrices can be inverted (by checking whether all
        determinants are >= =0.) """

        # Megacam
        if todo['WL']:
            massModel_var = self.WLcalib['MegacamSim'][1]**2 + self.WLcalib['MegacamMcErr']**2 + self.WLcalib['MegacamCenterErr']**2
            zDistShear_var = self.WLcalib['MegacamzDistErr']**2 + self.WLcalib['MegacamShearErr']**2 + self.WLcalib['MegacamContamCorr']**2
            bias_std = msqrt(massModel_var + zDistShear_var)
            # bias = bSim + bMassModel + (bN(z)+bShearCal)
            # scaling['bWL_Megacam'] = self.WLcalib['MegacamSim'][0] + scaling['WLbias']*massModelErr + scaling['MegacamBias']*zDistShearErr
            scaling['bWL_Megacam'] = self.WLcalib['MegacamSim'][0] + scaling['MegacamBias']*bias_std
            # lognormal scatter
            scaling['DWL_Megacam'] = self.WLcalib['MegacamSim'][2] + scaling['WLscatter']*self.WLcalib['MegacamSim'][3]

            # HST
            zDistShear_var = self.HSTcalib['shape_unc']**2 + self.HSTcalib['zdist_unc']**2
            mass_model_var = self.HSTcalib['bias_unc']**2 + self.HSTcalib['Mc_unc']**2 + self.HSTcalib['miscent_unc']**2
            bias_std = np.sqrt(zDistShear_var + mass_model_var)
            scaling['bWL_HST'], scaling['DWL_HST'] = {}, {}
            for n,name in enumerate(self.HSTcalib['SPT_ID']):
                # bias = bSim + bMassModel + (bN(z)+bShearCal)
                scaling['bWL_HST'][name] = self.HSTcalib['bias'][n] + scaling['HSTbias']*bias_std[n]
                # lognormal scatter
                DWL_HST = self.HSTcalib['scatter'][n] + scaling['WLscatter']*self.HSTcalib['scatter_unc'][n]
                scaling['DWL_HST'][name] = DWL_HST
                # SZ WL covariance matrix
                cov = [[DWL_HST**2, scaling['rhoSZWL']*scaling['Dsz']*DWL_HST],
                       [scaling['rhoSZWL']*scaling['Dsz']*DWL_HST, scaling['Dsz']**2]]
                if np.linalg.det(cov) <= 0.:
                    return False
                scaling['cov_HST_SZ_%s' % name] = np.array(cov)
                # SZ WL X covariance matrix
                if todo['Yx'] or todo['Mgas']:
                    cov = [[DWL_HST**2, scaling['rhoWLX']*DWL_HST*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*DWL_HST],
                           [scaling['rhoWLX']*DWL_HST*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
                           [scaling['rhoSZWL']*scaling['Dsz']*DWL_HST, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
                    if np.linalg.det(cov) <= 0.:
                        return False
                    scaling['cov_HST_X_SZ_%s' % name] = np.array(cov)
                # SZ WL richness covariance matrix
                if todo['richness']:
                    cov = [[DWL_HST**2, scaling['rhoWLrichness']*DWL_HST*scaling['Drichness'], scaling['rhoSZWL']*scaling['Dsz']*DWL_HST],
                           [scaling['rhoWLrichness']*DWL_HST*scaling['Drichness'], scaling['Drichness']**2, scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness']],
                           [scaling['rhoSZWL']*scaling['Dsz']*DWL_HST, scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness'], scaling['Dsz']**2]]
                    if np.linalg.det(cov) <= 0.:
                        return False
                    scaling['cov_HST_richness_SZ_%s' % name] = np.array(cov)

        # X-ray
        if todo['Yx'] or todo['Mgas']:
            cov = [[scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
                   [scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
            if np.linalg.det(cov) <= 0.:
                return False
            scaling['cov_X_SZ'] = np.array(cov)

        # Richness
        if todo['richness']:
            cov = [[scaling['Drichness']**2, scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness']],
                   [scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness'], scaling['Dsz']**2]]
            if np.linalg.det(cov) <=0.:
                return False
            scaling['cov_richness_SZ'] = np.array(cov)

        # WL: Megacam
        if todo['WL']:
            cov = [[scaling['DWL_Megacam']**2, scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam']],
                   [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam'], scaling['Dsz']**2]]
            if np.linalg.det(cov) <=0.:
                return False
            scaling['cov_Megacam_SZ'] = np.array(cov)

        # WL: DES
        if 'DES_m_piv' in scaling.keys():
            z = np.array([.25, .25, .85, .85])
            M = np.array([1e13, 1e16, 1e13, 1e16])
            DES_scatter = scaling_relations.WLscatter('WLDES', np.log(M), z, scaling)
            dets = [np.linalg.det([[DES_scatter[i]**2, scaling['rhoSZWL']*scaling['Dsz']*DES_scatter[i]],
                                   [scaling['rhoSZWL']*scaling['Dsz']*DES_scatter[i], scaling['Dsz']**2]])
                    for i in range(4)]
            if np.any(np.array(dets)<=0.):
                return False

        # X-ray and WL: Megacam
        if todo['WL'] and (todo['Yx'] or todo['Mgas']):
            cov = [[scaling['DWL_Megacam']**2, scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam']],
                   [scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
                   [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam'], scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
            if np.linalg.det(cov) <= 0.:
                return False
            scaling['cov_Megacam_X_SZ'] = np.array(cov)

        # DES WL and Richness [DES WL, richness, SZ]
        if todo['WL'] and todo['richness']:
            if 'DES_m_piv' in scaling.keys():
                z = np.array([.25, .25, .85, .85])
                M = np.array([1e13, 1e16, 1e13, 1e16])
                DES_scatter = scaling_relations.WLscatter('WLDES', np.log(M), z, scaling)
                dets = [np.linalg.det([[DES_scatter[i]**2, scaling['rhoWLrichness']*DES_scatter[i]*scaling['Drichness'], scaling['rhoSZWL']*DES_scatter[i]*scaling['Dsz']],
                                       [scaling['rhoWLrichness']*DES_scatter[i]*scaling['Drichness'], scaling['Drichness']**2, scaling['rhoSZrichness']*scaling['Drichness']*scaling['Dsz']],
                                       [scaling['rhoSZWL']*DES_scatter[i]*scaling['Dsz'], scaling['rhoSZrichness']*scaling['Drichness']*scaling['Dsz'], scaling['Dsz']**2]])
                        for i in range(4)]
                if np.any(np.array(dets) <= 0.):
                    return False

        return True
