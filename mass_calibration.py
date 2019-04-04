from __future__ import division
import numpy as np
import os
import imp
from multiprocessing import Pool
from astropy.table import Table

import scipy.special as ss
from scipy import integrate, signal
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.stats import norm, lognorm, multivariate_normal

import cosmo, lensing, Mconversion_concentration, scaling_relations

cosmologyRef = {'Omega_m':.272, 'Omega_l':.728, 'h':.702, 'w0':-1, 'wa':0}
GETPULL = False

# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return MassCalibration.clusterlike(*arg)

################################################################################
class MassCalibration:

    def __init__(self, todo, scaling, mcType, surveyCutSZ, surveyCutRedshift,
                 SPT_survey_fields, SPT_doublecounts, SPTcatalogfile,
                 observable_pairs,
                 WLsimcalibfile, DES_betabias_file, HSTfile, MegacamFile, DESfile,
                 NPROC):

        self.NPROC = NPROC
        self.todo = todo
        self.scaling = scaling
        self.mcType = mcType
        self.surveyCutSZ = surveyCutSZ
        self.surveyCutRedshift = surveyCutRedshift
        self.observable_pairs = observable_pairs

        # Read input files
        self.SPT_survey = Table.read(SPT_survey_fields, format='ascii.commented_header')
        SPTdata = imp.load_source('SPTdata', SPT_doublecounts)
        self.SPTdoubleCount = SPTdata.SPTdoubleCount
        self.catalog = Table.read(SPTcatalogfile)
        WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
        self.WLcalib = WLsimcalib.WLcalibration
        if self.todo['WL']:
            self.WL = lensing.SPTlensing(self.catalog, WLsimcalibfile,
                                         HSTfile, MegacamFile, DESfile, DES_betabias_file)

        self.HMF_convo_names = [['Yx', 'Yx_SZ'],
                                ['Mgas', 'Mgas_SZ'],
                                ['WLMegacam', 'Megacam_SZ'],
                                ['WLDES', 'DES_SZ'],
                                ['WLHST', 'HST_SZ'],
                                ['richness', 'richness_SZ'],
                                [['WLMegacam', 'Yx'], 'Megacam_Yx_SZ'],
                                [['WLDES', 'Yx'], 'DES_Yx_SZ'],
                                [['WLMegacam', 'Mgas'], 'Megacam_Mgas_SZ'],
                                [['WLDES', 'Mgas'], 'DES_Mgas_SZ'],]



    ############################################################################
    def lnlike(self):
        """Returns ln-likelihood for mass calibration of the whole cluster sample."""
        ##### WL: Precompute array of angular diameter distances
        if self.todo['WL']:
            self.WL.get_dAs(self.cosmology)

        ##### Set up interpolation for HMF
        HMF_in = self.HMF['dNdlnM'][1:,:]
        if np.any(HMF_in==0):
            HMF_in[np.where(HMF_in==0)] = np.nextafter(0, 1)
        self.HMF_interp = RectBivariateSpline(np.log(self.HMF['z_arr'][1:]), np.log(self.HMF['M_arr']), np.log(HMF_in))

        ##### Initialize mass-concentration relation class (for WL and dispersions)
        if self.todo['WL'] or self.todo['veldisp']:
            self.MCrel = Mconversion_concentration.ConcentrationConversion(self.mcType, self.cosmology)

        ##### Compute interpolation table for M500-M200
        if self.todo['WL'] or self.todo['veldisp']:
            z_arr = np.linspace(.1, 2, 20)
            M500 = np.logspace(np.log10(self.HMF['M_arr'][0]), np.log10(self.HMF['M_arr'][-1]), 20)
            M200 = np.array([np.array([self.MCrel.MDelta_to_M200(m, 500., z) for m in M500]) for z in z_arr])
            self.lnM500_to_lnM200 = RectBivariateSpline(z_arr, np.log(M500), np.log(M200))
        if self.todo['veldisp']:
            z_arr = np.linspace(.1, 2, 20)
            M200 = np.logspace(np.log10(np.amin(M200)), np.log10(np.amax(M200)), 20)
            M500 = np.array([np.array([self.MCrel.M200_to_MDelta(m, 500., z) for m in M200]) for z in z_arr])
            self.lnM200_to_lnM500 = RectBivariateSpline(z_arr, np.log(M200), np.log(M500))

        ##### Evaluate the individual likelihoods
        len_data = len(self.catalog['SPT_ID'])

        if self.NPROC==0:
            # Iterate through cluster list
            likelihoods = np.array([self.clusterlike(i) for i in range(len_data)])
        else:
            # Launch a multiprocessing pool and get the likelihoods
            pool = Pool(processes=self.NPROC)
            argin = zip([self]*len_data, range(len_data))
            likelihoods = pool.map(unwrap_self_f, argin)
            pool.close()

        # If likelihood computation failed it returned 0
        if np.count_nonzero(likelihoods)<len_data:
            return -np.inf

        lnlike = np.sum(np.log(likelihoods))

        return lnlike



    ############################################################################
    def clusterlike(self, i):
        """Return multi-wavelength mass-calibration likelihood (no log!) for a
        given cluster (index) by calling get_P_1obs_xi or get_P_2obs_xi or
        returning 1 if no follow-up data is available."""
        name = self.catalog['SPT_ID'][i]

        ##### Do we actually want this guy? (some clusters in SPT-SZ are at field boundaries)
        if (name,self.catalog['FIELD'][i]) in self.SPTdoubleCount:
            return 1.
        if not self.surveyCutSZ[0]<self.catalog['XI'][i]<self.surveyCutSZ[1] or not self.surveyCutRedshift[0]<self.catalog['REDSHIFT'][i]<self.surveyCutRedshift[1]:
            return 1

        ##### Check if follow-up is available
        nobs = 0
        obsnames = []
        if self.todo['WL'] and self.catalog['WLdata'][i] is not None:
            nobs+= 1
            if self.catalog['WLdata'][i]['datatype']=='Megacam':
                obsnames.append('WLMegacam')
            elif self.catalog['WLdata'][i]['datatype']=='DES':
                obsnames.append('WLDES')
            elif self.catalog['WLdata'][i]['datatype']=='HST':
                obsnames.append('WLHST')
        if self.todo['veldisp'] and self.catalog['veldisp'][i]!=0.:
            nobs+= 1
            obsnames.append('disp')
        if self.todo['Yx'] and self.catalog['Mg_fid'][i]!=0:
            nobs+= 1
            obsnames.append('Yx')
        if self.todo['Mgas'] and self.catalog['Mg_fid'][i]!=0:
            nobs+= 1
            obsnames.append('Mgas')
        if self.todo['richness'] and self.catalog['LAMBDA_RM'][i]!=0.:
            nobs+= 1
            obsnames.append('richness')
        if nobs==0:
            return 1.

        ##### Set SPT field scaling factor
        self.thisSPTfield_gamma = self.SPT_survey['GAMMA'][self.SPT_survey['FIELD']==self.catalog['FIELD'][i]]

        #####
        if nobs==1:
            # Get the name of the multi-obs HMF
            for obs in self.HMF_convo_names:
                if obsnames[0]==obs[0]:
                    pair_name = obs[1]

            probability = self.get_P_1obs_xi(obsnames[0], i, pair_name)

        elif nobs==2:
            # Get the name of the multi-obs HMF
            for obs in self.HMF_convo_names:
                if obsnames==obs[0]:
                    pair_name = obs[1]

            probability = self.get_P_2obs_xi(obsnames, i, pairname)

        else:
            raise ValueError(name,"has",nobs,"follow-up observables. I don't know what to do!")

        if (probability<0) | (np.isnan(probability)):
            return 0
            # raise ValueError("P(obs|xi) =", probability, name)

        # print name, obsnames, probability
        return probability


    ############################################################################
    def conversion_factor_Xray_obs_r500ref(self, redshift):
        """Account for the cosmological dependence of the X-ray observable and
        convert to the model expectation at r500ref using the slope of the
        radial profile. This is done for the mass array self.HMF['M_arr']."""
        # Angular diameter distances in current and reference cosmology [Mpc]
        dA = cosmo.dA(redshift, self.cosmology)/self.cosmology['h']
        dAref = cosmo.dA(redshift, cosmologyRef)/cosmologyRef['h']
        # R500 [kpc]
        rho_c_z = cosmo.RHOCRIT * cosmo.Ez(redshift, self.cosmology)**2
        r500 = 1000 * (3*self.HMF['M_arr']/(4*np.pi*500*rho_c_z))**(1/3) / self.cosmology['h']
        # r500 in reference cosmology [kpc]
        r500ref = r500 * dAref/dA
        # Xray observable at fiducial r500...
        correction = (self.catalog['r500'][dataID]/r500ref)**self.scaling['dlnMg_dlnr']
        # ... corrected to reference cosmology
        correction*= (dAref/dA)**2.5

        return correction


    def get_multiobs_HMF_z(self, z, z_arr, lnHMF):
        """Interpolate HMF[z, obs_0...N] to redshift z using linear
        interpolation of z_arr in log-log space."""
        lnz_arr = np.log(z_arr)
        idx_lo = np.where(z_arr<z)[0][-1]
        Delta_lnz = np.log(z)-lnz_arr[idx_lo]
        Delta_lny = lnHMF[idx_lo+1]-lnHMF[idx_lo]
        res = np.exp(lnHMF[idx_lo] + Delta_lnz*Delta_lny)
        return res


    def convolve_HMF_lnobs_to_xi(self, xi, zeta_arr, xi_arr, HMF):
        """Return P(ln(multi-obs) | xi). Start from multi-obs `HMF[ln(obs0),
        ln(obs1), ..., ln(zeta)]`, set elements with zeta<2 to 0, convolve with
        unit variance in xi and evaluate at `xi`."""
        shape = HMF.shape
        if len(shape)==2:
            this_xi_arr = xi_arr[None,:]
            HMF[:,np.where(zeta_arr<2)] = 0
        elif len(shape)==3:
            this_xi_arr = xi_arr[None,None,:]
            HMF[:,:,np.where(zeta_arr<2)] = 0
        # dP/dxi = dP/dlnzeta dlnzeta/dxi
        HMF_xi = HMF * scaling_relations.dlnzeta_dxi(this_xi_arr)
        # Simultaneous convolution and evaluation at xi
        unit_var_kernel = norm.pdf(xi, this_xi_arr, 1)
        HMF_at_xi = np.trapz(HMF_xi * unit_var_kernel, this_xi_arr, axis=-1)
        return HMF_at_xi


    def apply_sys_Poisson_scatter_richness(self, obs_arr, lnobs_arr, dP_dlnobs):
        """Convolve dP/dlnlambda with lognormal scatter of width var=1/lambda.
        This mimics the Poisson error on counting member galaxies."""
        integrand = dP_dlnobs[None,:] * norm.pdf(lnobs_arr[:,None], lnobs_arr[None,:], 1/obs_arr[None,:]**.5)
        dP_dlnobs = np.trapz(integrand, lnobs_arr, axis=1)
        return dP_dlnobs


    def convolve_WL_LSS(self, obs_arr, dP_dobs, LSSnoise):
        """Convolve dP/dMwl with Gaussian scatter to account for noise by
        large-scale structure."""
        integrand = dP_dobs[None,:] * norm.pdf(obs_arr[:,None], obs_arr[None,:], LSSnoise)
        dP_dobs = np.trapz(integrand, obs_arr, axis=1)
        # Normalize to be sure
        dP_dobs/= np.trapz(dP_dobs, obs_arr)
        return dP_dobs


    def downsample_distribution(self, x, y, N_target=48):
        idx_zero = np.where(y>0)[0]
        x_out = x[idx_zero]
        y_out = x[idx_zero]
        if len(x_out)<=N_target:
            return x_out, y_out

        y_interp = interp1d(x_out, y_out, kind='cubic')

        mean = np.trapz(x_out*y_out, x_out)
        std = np.sqrt(np.trapz((y_out-mean)**2, x_out))
        x_min = np.amax((x_out[0], mean-4*std))
        x_max = np.amin((x_out[-1], mean+4*std))
        x_new = np.linspace(x_min, x_max, N_target)
        y_new = y_interp(x_new)

        return x_new, y_new        


    ############################################################################
    def get_P_1obs_xi(self, obsname, dataID, pairname):
        """Returns P(obs|xi,z,p) for a single type of follow-up data."""

        ##### Get the follow-up observable, obsintr is used for setting up mass range
        if obsname=='Yx':
            obsmeas, obserr = self.catalog['Yx_fid'][dataID], self.catalog['Yx_err'][dataID]
        elif obsname=='Mgas':
            obsmeas, obserr = self.catalog['Mg_fid'][dataID], self.catalog['Mg_err'][dataID]
        elif obsname=='disp':
            obsmeas, obserr = self.catalog['veldisp'][dataID], self.scaling['DdispN']/self.catalog['Ngal'][dataID]
        elif obsname=='richness':
            obsmeas, obserr = self.catalog['LAMBDA_RM'][dataID], self.catalog['LAMBDA_RM_UNC'][dataID]
        elif obsname=='WLMegacam':
            LSSnoise = self.WLcalib['Megacam_LSS'][0] + self.scaling['MegacamScatterLSS'] * self.WLcalib['Megacam_LSS'][1]
        elif obsname=='WLHST':
            LSSnoise = self.WLcalib['HST_LSS'][0] + self.scaling['HSTscatterLSS'] * self.WLcalib['HST_LSS'][1]
        elif obsname=='WLDES':
            LSSnoise = self.WLcalib['DES_LSS'][0] + self.scaling['DESscatterLSS'] * self.WLcalib['DES_LSS'][1]

        ##### Observable arrays
        zeta_arr = self.thisSPTfield_gamma * scaling_relations.mass2obs('zeta', self.HMF['M_arr'], self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology)
        lnzeta_arr = np.log(zeta_arr)
        xi_arr = scaling_relations.zeta2xi(zeta_arr)
        obsArr = scaling_relations.mass2obs(obsname, self.HMF['M_arr'], self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology)
        # Account for radial dependence for X-ray observables
        if obsname in ('Mgas', 'Yx'):
            correction = self.conversion_factor_Xray_obs_r500ref(self.catalog['REDSHIFT'][dataID])
            obsArr*= correction
        lnobsArr = np.log(obsArr)

        ##### dN/dlnobs/dlnzeta at z=z_cluster from interpolation tables
        HMF_2d = self.get_multiobs_HMF_z(z=self.catalog['REDSHIFT'][dataID],
                                         z_arr=self.HMF_convos['%s_z'%pairname],
                                         lnHMF=self.HMF_convos[pairname])

        ##### P(ln(obs) | xi)
        dP_dlnobs = self.convolve_HMF_lnobs_to_xi(self.catalog['XI'][dataID], zeta_arr, xi_arr, HMF_2d)

        #### Convolve with additional lognormal scatter in richness
        if obsname=='richness':
            dP_dlnobs = self.apply_sys_Poisson_scatter_richness(obsArr, lnobsArr, dP_dlnobs)

        #### Go to linear obs space and normalize
        # dP/dobs = dP/dlnobs * dlnobs/dobs = dP/dlnobs /obs
        dP_dobs = dP_dlnobs/obsArr
        dP_dobs/= np.trapz(dP_dobs, obsArr)


        ##### Evaluate likelihood
        if obsname in ('Yx', 'Mgas', 'richness'):
            likeli = np.trapz(dP_dobs*norm.pdf(obsmeas, obsArr, obserr), obsArr)

            if GETPULL:
                integrand = dP_dobs[None,:] * norm.pdf(obsArr[:,None], obsArr[None,:], obserr)
                dP_dobs_obs = np.trapz(integrand, obsArr, axis=1)
                dP_dobs_obs/= np.trapz(dP_dobs_obs,obsArr)
                cumtrapz = integrate.cumtrapz(dP_dobs_obs,obsArr)
                perc = np.interp(obsmeas, obsArr[1:], cumtrapz)
                print '%s %.4f %.4f %.4f %.4e'%(
                    self.catalog['SPT_ID'][dataID], self.catalog['XI'][dataID], self.catalog['REDSHIFT'][dataID], obsmeas, 2**.5 * ss.erfinv(2*perc-1))

        elif obsname=='disp':
            dP_dobs_meas = lognorm.pdf(obsmeas, scale=obsArr, s=obserr)
            likeli = np.trapz(dP_dobs*dP_dobs_meas, obsArr)

        elif obsname in ('WLHST', 'WLMegacam', 'WLDES'):
            # Convolve with Gaussian LSS scatter
            if LSSnoise>0.:
                dP_dobs = self.convolve_WL_LSS(obsArr, dP_dobs, LSSnoise)
            obsArr, dP_dobs = self.downsample_distribution(obsArr, dP_dobs)
            # P(Mwl) from data
            Pwl = self.WL.like(self.catalog, dataID, obsArr, self.cosmology, self.MCrel, self.lnM500_to_lnM200, self.scaling)
            # Get likelihood
            likeli = np.trapz(Pwl*dP_dobs, obsArr)


        if ((likeli<0)|(np.isnan(likeli))):
            print self.catalog['SPT_ID'][dataID], obsname, likeli
            #np.savetxt(self.catalog['SPT_ID'][dataID],np.transpose((obsArr, dP_dobs)))
            return 0.

        return likeli



    ############################################################################
    def get_P_2obs_xi(self, obsnames, dataID, pairname):
        """Returns P(obs1, obs2|xi,z,p) for two types of follow-up data (e.g.,
        WL and X-ray)."""

        ##### Get observables, obsintr is used for setting up mass range
        obsmeas, obserr = np.empty(2), np.empty(2)
        for i in range(2):
            if obsnames[i]=='Yx':
                obsmeas[i], obserr[i] = self.catalog['Yx_fid'][dataID], self.catalog['Yx_err'][dataID]
            elif obsnames[i]=='Mgas':
                obsmeas[i], obserr[i] = self.catalog['Mg_fid'][dataID], self.catalog['Mg_err'][dataID]
            elif obsnames[i]=='disp':
                obsmeas[i], obserr[i] = self.catalog['veldisp'][dataID], self.scaling['DdispN']/self.catalog['Ngal'][dataID]
            elif obsnames[i]=='richness':
                obsmeas[i], obserr[i] = self.catalog['richness'][dataID], self.catalog['richness_err'][dataID]
            elif obsnames[i]=='WLMegacam':
                LSSnoise = self.WLcalib['Megacam_LSS'][0] + self.scaling['MegacamScatterLSS'] * self.WLcalib['Megacam_LSS'][1]
            elif obsnames[i]=='WLHST':
                LSSnoise = self.WLcalib['HST_LSS'][0] + self.scaling['HSTscatterLSS'] * self.WLcalib['HST_LSS'][1]
            elif obsnames[i]=='WLDES':
                LSSnoise = self.WLcalib['DES_LSS'][0] + self.scaling['DESscatterLSS'] * self.WLcalib['DES_LSS'][1]

        ##### Observable arrays
        zeta_arr = self.thisSPTfield_gamma * scaling_relations.mass2obs('zeta', self.HMF['M_arr'], self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology)
        lnzeta_arr = np.log(zeta_arr)
        xi_arr = scaling_relations.zeta2xi(zeta_arr)
        obsArr, lnobsArr = [], []
        for i in range(2):
            obsArrTemp = scaling_relations.mass2obs(obsnames[i], self.HMF['M_arr'], self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology)
            # Account for radial dependence for X-ray observables
            if obsnames[i] in ('Mgas', 'Yx'):
                correction = self.conversion_factor_Xray_obs_r500ref(self.catalog['REDSHIFT'][dataID])
                obsArrTemp*= correction
            obsArr.append( obsArrTemp )
            lnobsArr.append( np.log(obsArrTemp) )

        ##### dN/dlnobs0/dlnobs1/dlnzeta at z=z_cluster from interpolation tables
        HMF_3d = self.get_multiobs_HMF_z(z=self.catalog['REDSHIFT'][dataID],
                                         z_arr=self.HMF_convos['%s_z'%pairname],
                                         lnHMF=self.HMF_convos[pairname])

        ##### P(ln(obs0, obs1) | xi)
        dP_dlnobs = self.convolve_HMF_lnobs_to_xi(self.catalog['XI'][dataID], zeta_arr, xi_arr, HMF_3d)

        ##### Go to linear space [obs0][obs1]
        dP_dobs01 = dP_dlnobs/obsArr[0][:,None]/obsArr[1][None,:]

        ##### P0
        dP_dobs0 = np.trapz(dP_dobs01, obsArr[1], axis=1)
        dP_dobs0/= np.trapz(dP_dobs0, obsArr[0])

        if obsnames[0] in ('WLHST', 'WLMegacam', 'WLDES'):
            # Convolve with Gaussian LSS scatter
            if LSSnoise>0.:
                dP_dobs0 = self.convolve_WL_LSS(obsArr[0], dP_dobs0, LSSnoise)
            # P(Mwl) from data
            Pobs = self.WL.like(self.catalog, dataID, obsArr[0], self.cosmology, self.MCrel, self.lnM500_to_lnM200, self.scaling)
        else: print "not ready!"

        likeli0 = np.trapz(dP_dobs0*Pobs, obsArr[0])

        ##### P1 (Yx)
        dP_dobs1 = np.trapz(dP_dobs01, obsArr[0], axis=0)

        # Normalize (in principe, multiply with dlnX/dlnXfid, but this is mass-independent)
        dP_dobs1/= np.trapz(dP_dobs1, obsArr[1])
        likeli1 = np.trapz(dP_dobs1*norm.pdf(obsmeas[1], obsArr[1], obserr[1]), obsArr[1])


        ##### Probability
        likeli = likeli0*likeli1

        return likeli
