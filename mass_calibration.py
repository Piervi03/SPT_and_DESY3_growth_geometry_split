import numpy as np
import imp
from multiprocessing import Pool
from astropy.table import Table
# import time

import scipy.special as ss
from scipy import integrate
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.stats import norm, lognorm

import cosmo
import Mconversion_concentration
import scaling_relations

cosmologyRef = {'Omega_m': .272, 'Omega_l': .728, 'h': .702, 'w0': -1, 'wa': 0}
GETPULL = False


# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return MassCalibration.clusterlike(*arg)

################################################################################


class MassCalibration:

    def __init__(self, todo, mcType,
                 z_cl_min_max, lambda_min,
                 SPT_survey_fields, SPTcatalogfile,
                 observable_pairs,
                 WLsimcalibfile,
                 NPROC):

        self.NPROC = NPROC
        self.todo = todo
        self.mcType = mcType
        self.z_cl_min_max = z_cl_min_max
        self.lambda_min = lambda_min
        self.observable_pairs = observable_pairs

        # Read input files
        self.SPT_survey = Table.read(SPT_survey_fields, format='ascii.commented_header')
        self.catalog = Table.read(SPTcatalogfile)
        WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
        self.WLcalib = WLsimcalib.WLcalibration

        self.HMF_convo_names = [['Yx', 'Yx_SZ'],
                                ['Mgas', 'Mgas_SZ'],
                                ['WLMegacam', 'Megacam_SZ'],
                                ['WLDES', 'DES_SZ'],
                                ['WLHST', 'HST_SZ'],
                                ['richness', 'richness_SZ'],
                                [['WLMegacam', 'Yx'], 'Megacam_Yx_SZ'],
                                [['WLDES', 'Yx'], 'DES_Yx_SZ'],
                                [['WLMegacam', 'Mgas'], 'Megacam_Mgas_SZ'],
                                [['WLDES', 'Mgas'], 'DES_Mgas_SZ'],
                                [['WLDES', 'richness'], 'DES_richness_SZ'],
                                ]

    ############################################################################

    def lnlike(self, HMF_convos, cosmology, scaling):
        """Returns ln-likelihood for mass calibration of the whole cluster sample."""
        self.HMF_convos = HMF_convos
        self.cosmology = cosmology
        self.scaling = scaling
        self.xi_min = scaling_relations.zeta2xi(self.scaling['zeta_min'])

        # Initialize mass-concentration relation class (for WL and dispersions)
        if self.todo['veldisp']:
            self.MCrel = Mconversion_concentration.ConcentrationConversion(self.mcType, self.cosmology,
                                                                           setup_interp=True, interp_massdef=500)

        # Evaluate the individual likelihoods
        len_data = len(self.catalog['SPT_ID'])

        if self.NPROC == 0:
            # Iterate through cluster list
            likelihoods = np.array([self.clusterlike(i) for i in range(len_data)])
        else:
            # Launch a multiprocessing pool and get the likelihoods
            with Pool(processes=self.NPROC) as pool:
                argin = zip([self]*len_data, range(len_data))
                likelihoods = pool.map(unwrap_self_f, argin)

        # If likelihood computation failed it returned 0
        if np.count_nonzero(likelihoods) < len_data:
            return -np.inf

        lnlike = np.sum(np.log(likelihoods))

        return lnlike

    ############################################################################

    def clusterlike(self, i):
        """Return multi-wavelength mass-calibration likelihood (no log!) for a
        given cluster (index) by calling get_P_1obs_xi or get_P_2obs_xi or
        returning 1 if no follow-up data is available."""
        # t0 = time.time()
        name = self.catalog['SPT_ID'][i]

        # Do we actually want this guy?
        if not self.SPT_survey['XI_MIN'][self.SPT_survey['FIELD'] == self.catalog['FIELD'][i]] < self.catalog['XI'][i] or not self.z_cl_min_max[0] < self.catalog['REDSHIFT'][i] < self.z_cl_min_max[1]:
            return 1

        # Check if follow-up is available
        nobs = 0
        obsnames = []
        if self.todo['WL'] and self.catalog['WLdata'][i] is not None:
            nobs += 1
            if self.catalog['WLdata'][i]['datatype'] == 'Megacam':
                obsnames.append('WLMegacam')
            elif self.catalog['WLdata'][i]['datatype'] == 'DES':
                obsnames.append('WLDES')
            elif self.catalog['WLdata'][i]['datatype'] == 'HST':
                obsnames.append('WLHST')
        if self.todo['veldisp'] and self.catalog['veldisp'][i] != 0.:
            nobs += 1
            obsnames.append('disp')
        if self.todo['Yx'] and self.catalog['Mg_fid'][i] != 0:
            nobs += 1
            obsnames.append('Yx')
        if self.todo['Mgas'] and self.catalog['Mg_fid'][i] != 0:
            nobs += 1
            obsnames.append('Mgas')
        if self.todo['richness'] and self.catalog['richness'][i] != 0.:
            nobs += 1
            obsnames.append('richness')
        if nobs == 0:
            return 1.

        #####
        if nobs == 1:
            # Get the name of the multi-obs HMF
            for obs in self.HMF_convo_names:
                if obsnames[0] == obs[0]:
                    pair_name = obs[1]

            probability = self.get_P_1obs_xi(obsnames[0], i, pair_name)

        elif nobs == 2:
            # Get the name of the multi-obs HMF
            for obs in self.HMF_convo_names:
                if obsnames == obs[0]:
                    pair_name = obs[1]

            probability = self.get_P_2obs_xi(obsnames, i, pair_name)

        else:
            raise ValueError(name, "has", nobs, "follow-up observables. I don't know what to do!")

        if (probability < 0) | (np.isnan(probability)):
            return 0
            # raise ValueError("P(obs|xi) =", probability, name)

        # print(name, obsnames, probability)#, time.time()-t0)
        return probability

    ############################################################################

    def conversion_factor_Xray_obs_r500ref(self, dataID):
        """Account for the cosmological dependence of the X-ray observable and
        convert to the model expectation at r500ref using the slope of the
        radial profile. This is done for the mass array self.HMF_convos['M_arr']."""
        # Angular diameter distances in current and reference cosmology [Mpc]
        dA = cosmo.dA(self.catalog['REDSHIFT'][dataID], self.cosmology)/self.cosmology['h']
        dAref = cosmo.dA(self.catalog['REDSHIFT'][dataID], cosmologyRef)/cosmologyRef['h']
        # R500 [kpc]
        rho_c_z = cosmo.RHOCRIT * cosmo.Ez(self.catalog['REDSHIFT'][dataID], self.cosmology)**2
        r500 = 1000 * (3*self.HMF_convos['M_arr']/(4*np.pi*500*rho_c_z))**(1/3) / self.cosmology['h']
        # r500 in reference cosmology [kpc]
        r500ref = r500 * dAref/dA
        # Xray observable at fiducial r500...
        correction = (self.catalog['r500'][dataID]/r500ref)**self.scaling['dlnMg_dlnr']
        # ... corrected to reference cosmology
        correction *= (dAref/dA)**2.5

        return correction

    def get_multiobs_lnHMF_z(self, z, z_arr, lnHMF):
        """Interpolate HMF[z, obs_0...N] to redshift z using linear
        interpolation of z_arr in log-log space."""
        idx_lo = np.digitize(z, z_arr)-1
        Delta_lnz = np.log(z/z_arr[idx_lo]) / np.log(z_arr[idx_lo+1]/z_arr[idx_lo])
        # We know we'll get NANs because we're potentially doing -inf -(-inf)
        with np.errstate(invalid='ignore'):
            Delta_lny = lnHMF[idx_lo+1]-lnHMF[idx_lo]
            res = lnHMF[idx_lo] + Delta_lnz*Delta_lny
        # Remove those NANs
        res[np.isnan(res)] = -np.inf
        return res

    def convolve_HMF_lnobs_to_xi(self, xi, xi_arr, ln_HMF):
        """Return P(ln(multi-obs) | xi). Start from multi-obs `HMF[ln(obs0),
        ln(obs1), ..., ln(zeta)]`, set elements with zeta<2 to 0, convolve with
        unit variance in xi and evaluate at `xi`."""
        # We only need the whole thing "close" to xi (+4/-3 sigma)
        xi_lo = np.amax((self.xi_min, xi-4))
        xi_hi = np.amin((xi_arr[-1]-.01, xi+3))
        xi_arr_int = np.linspace(xi_lo, xi_hi, 32)
        # Interpolation in ln(xi)
        idx_lo = np.digitize(xi_arr_int, xi_arr)-1
        Delta_lnxi = np.log(xi_arr_int/xi_arr[idx_lo]) / np.log(xi_arr[idx_lo+1]/xi_arr[idx_lo])
        shape = ln_HMF.shape
        # We know we'll get NANs because we're potentially doing -inf -(-inf)
        with np.errstate(invalid='ignore'):
            if len(shape) == 2:
                Delta_lnH = ln_HMF[:, idx_lo+1]-ln_HMF[:, idx_lo]
                HMF_integrand = np.exp(ln_HMF[:, idx_lo] + Delta_lnxi*Delta_lnH)
                this_xi_arr = xi_arr_int[None, :]
            elif len(shape) == 3:
                Delta_lnH = ln_HMF[:, :, idx_lo+1]-ln_HMF[:, :, idx_lo]
                HMF_integrand = np.exp(ln_HMF[:, :, idx_lo] + Delta_lnxi*Delta_lnH)
                this_xi_arr = xi_arr_int[None, None, :]
            elif len(shape) == 4:
                Delta_lnH = ln_HMF[:, :, :, idx_lo+1]-ln_HMF[:, :, :, idx_lo]
                HMF_integrand = np.exp(ln_HMF[:, :, :, idx_lo] + Delta_lnxi*Delta_lnH)
                this_xi_arr = xi_arr_int[None, None, None, :]
            HMF_integrand[np.isnan(HMF_integrand)] = 0.
        # dP/dxi = dP/dlnzeta dlnzeta/dxi
        HMF_xi = HMF_integrand * scaling_relations.dlnzeta_dxi_given_xi(this_xi_arr)
        # Simultaneous convolution and evaluation at xi
        unit_var_kernel = norm.pdf(xi, this_xi_arr, 1)
        HMF_at_xi = np.trapezoid(HMF_xi * unit_var_kernel, this_xi_arr, axis=-1)
        return HMF_at_xi

    def apply_sys_Poisson_scatter_richness(self, obs_arr, lnobs_arr, dP_dlnobs):
        """Convolve dP/dlnlambda with lognormal scatter of width var=1/lambda.
        This mimics the Poisson error on counting member galaxies."""
        integrand = dP_dlnobs[None, :] * norm.pdf(lnobs_arr[:, None], lnobs_arr[None, :], 1/np.sqrt(obs_arr[None, :]))
        dP_dlnobs = np.trapezoid(integrand, lnobs_arr, axis=1)
        return dP_dlnobs

    def convolve_WL_LSS(self, obs_arr, dP_dobs, LSSnoise):
        """Convolve dP/dMwl with Gaussian scatter to account for noise by
        large-scale structure."""
        integrand = dP_dobs[None, :] * norm.pdf(obs_arr[:, None], obs_arr[None, :], LSSnoise)
        dP_dobs = np.trapezoid(integrand, obs_arr, axis=1)
        # Normalize to be sure
        dP_dobs /= np.trapezoid(dP_dobs, obs_arr)
        return dP_dobs

    ############################################################################
    def get_P_1obs_xi(self, obsname, dataID, pairname):
        """Returns P(obs|xi,z,p) for a single type of follow-up data."""
        # dN/dlnobs/dlnzeta at z=z_cluster from interpolation tables
        if obsname == 'WLHST':
            lnHMF_2d = self.HMF_convos[pairname][self.catalog['SPT_ID'][dataID]]
        else:
            lnHMF_2d = self.get_multiobs_lnHMF_z(z=self.catalog['REDSHIFT'][dataID],
                                                 z_arr=self.HMF_convos['%s_z' % pairname],
                                                 lnHMF=self.HMF_convos[pairname])

        # Observable array
        obsArr = np.exp(scaling_relations.lnmass2lnobs(obsname, self.HMF_convos['lnM_arr'], self.catalog['REDSHIFT'][dataID],
                                                       self.scaling, self.cosmology, self.catalog['SPT_ID'][dataID]))
        # Account for radial dependence for X-ray observables
        if obsname in ('Mgas', 'Yx'):
            correction = self.conversion_factor_Xray_obs_r500ref(dataID)
            obsArr *= correction
        # Truncate at richness cut
        elif obsname == 'richness':
            if self.todo['lambda_min']:
                if self.catalog['FIELD'][dataID] == 'SPTPOL_500d':
                    this_lambda_min = self.lambda_min['deep'](self.catalog['REDSHIFT'][dataID])
                else:
                    this_lambda_min = self.lambda_min['shallow'](self.catalog['REDSHIFT'][dataID])
                lnHMF_2d_at_cut = [np.interp(this_lambda_min, obsArr, lnHMF_2d[:, i]) for i in range(lnHMF_2d.shape[1])]
                idx = obsArr > this_lambda_min
                obsArr = np.insert(obsArr[idx], 0, this_lambda_min)
                lnHMF_2d = np.insert(lnHMF_2d[idx, :], 0, lnHMF_2d_at_cut, axis=0)
        lnobsArr = np.log(obsArr)

        # SZ array
        zeta_arr = np.exp(scaling_relations.lnmass2lnobs('zeta', self.HMF_convos['lnM_arr'], self.catalog['REDSHIFT'][dataID],
                                                         self.scaling, self.cosmology,
                                                         SPTfield=self.SPT_survey[self.SPT_survey['FIELD'] == self.catalog['FIELD'][dataID]]))
        lnzeta_arr = np.log(zeta_arr)
        xi_arr = scaling_relations.zeta2xi(zeta_arr)
        if (xi_arr[0] > 2.7) | (self.catalog['XI'][dataID] > xi_arr[-1]-2):
            return 0

        # P(obs | xi)
        dP_dlnobs = self.convolve_HMF_lnobs_to_xi(self.catalog['XI'][dataID], xi_arr, lnHMF_2d)
        dP_dobs = dP_dlnobs/obsArr
        dP_dobs /= np.trapezoid(dP_dobs, obsArr)

        # Evaluate likelihood
        if obsname == 'richness':
            with np.errstate(divide='ignore'):
                lndP_dobs = np.log(dP_dobs)
            finite_idx = np.isfinite(lndP_dobs)
            lndP_dobs_interp = interp1d(obsArr[finite_idx], lndP_dobs[finite_idx],
                                        kind='linear', fill_value='extrapolate', assume_sorted=True)
            likeli = np.exp(lndP_dobs_interp(self.catalog['richness'][dataID]))

        elif obsname in ('Yx', 'Mgas'):
            if obsname == 'Yx':
                obsmeas, obserr = self.catalog['Yx_fid'][dataID], self.catalog['Yx_err'][dataID]
            elif obsname == 'Mgas':
                obsmeas, obserr = self.catalog['Mg_fid'][dataID], self.catalog['Mg_err'][dataID]

            likeli = np.trapezoid(dP_dobs*norm.pdf(obsmeas, obsArr, obserr), obsArr)

            if GETPULL:
                integrand = dP_dobs[None, :] * norm.pdf(obsArr[:, None], obsArr[None, :], obserr)
                dP_dobs_obs = np.trapezoid(integrand, obsArr, axis=1)
                dP_dobs_obs /= np.trapezoid(dP_dobs_obs, obsArr)
                cumtrapezoid = integrate.cumtrapezoid(dP_dobs_obs, obsArr)
                perc = np.interp(obsmeas, obsArr[1:], cumtrapezoid)
                print('%s %.4f %.4f %.4f %.4e' % (
                    self.catalog['SPT_ID'][dataID], self.catalog['XI'][dataID], self.catalog['REDSHIFT'][dataID],
                    obsmeas, 2**.5 * ss.erfinv(2*perc-1)))

        elif obsname == 'disp':
            obsmeas, obserr = self.catalog['veldisp'][dataID], self.scaling['DdispN']/self.catalog['Ngal'][dataID]
            dP_dobs_meas = lognorm.pdf(obsmeas, scale=obsArr, s=obserr)
            likeli = np.trapezoid(dP_dobs*dP_dobs_meas, obsArr)

        elif obsname in ('WLHST', 'WLMegacam', 'WLDES'):
            if obsname == 'WLMegacam':
                LSSnoise = self.WLcalib['Megacam_LSS'][0] + self.scaling['MegacamScatterLSS'] * self.WLcalib['Megacam_LSS'][1]
            elif obsname == 'WLHST':
                LSSnoise = self.WLcalib['HSTsim'][self.catalog['SPT_ID'][dataID]]['obs_scatter']
            elif obsname == 'WLDES':
                LSSnoise = 0.
            # Convolve with Gaussian LSS scatter
            if LSSnoise > 0.:
                dP_dobs = self.convolve_WL_LSS(obsArr, dP_dobs, LSSnoise)
            # P(Mwl) from data
            WL_interp = InterpolatedUnivariateSpline(np.log(self.WL.M_arr), self.catalog['lnp_Mwl'][dataID], k=1)
            # Get likelihood
            likeli = np.trapezoid(np.exp(WL_interp(lnobsArr))*dP_dobs, obsArr)

        if (likeli <= 0) | (np.isnan(likeli)) | (np.isinf(likeli)):
            print(self.catalog['SPT_ID'][dataID], obsname, likeli, np.amin(obsArr), np.amax(obsArr),)
            # np.savetxt(self.catalog['SPT_ID'][dataID],np.transpose((obsArr, lndP_dobs)))
            return 0.

        return likeli

    ############################################################################
    def get_P_2obs_xi(self, obsnames, dataID, pairname):
        """Returns P(obs1, obs2|xi,z,p) for two types of follow-up data (e.g.,
        WL and X-ray)."""
        # dN/dlnobs0/dlnobs1/dlnzeta at z=z_cluster from interpolation tables
        lnHMF_3d = self.get_multiobs_lnHMF_z(z=self.catalog['REDSHIFT'][dataID],
                                             z_arr=self.HMF_convos['%s_z' % pairname],
                                             lnHMF=self.HMF_convos[pairname])

        # Observable arrays
        obsArr, lnobsArr, obsmeas, obserr = [], [], np.empty(2), np.empty(2)
        for i in range(2):
            if obsnames[i] == 'Yx':
                obsmeas[i], obserr[i] = self.catalog['Yx_fid'][dataID], self.catalog['Yx_err'][dataID]
            elif obsnames[i] == 'Mgas':
                obsmeas[i], obserr[i] = self.catalog['Mg_fid'][dataID], self.catalog['Mg_err'][dataID]
            elif obsnames[i] == 'disp':
                obsmeas[i], obserr[i] = self.catalog['veldisp'][dataID], self.scaling['DdispN']/self.catalog['Ngal'][dataID]
            # elif obsnames[i] == 'richness':
            #    obsmeas[i], obserr[i] = self.catalog['LAMBDA_MCMF_COMB'][dataID], self.catalog['LAMBDA_RM_UNC'][dataID]
            elif obsnames[i] == 'WLMegacam':
                LSSnoise = self.WLcalib['Megacam_LSS'][0] + self.scaling['MegacamScatterLSS'] * self.WLcalib['Megacam_LSS'][1]
            elif obsnames[i] == 'WLHST':
                LSSnoise = self.WLcalib['HSTsim'][self.catalog['SPT_ID'][dataID]]['obs_scatter']
            elif obsnames[i] == 'WLDES':
                LSSnoise = 0.
            obsArrTemp = np.exp(scaling_relations.lnmass2lnobs(obsnames[i],
                                                               self.HMF_convos['lnM_arr'], self.catalog['REDSHIFT'][dataID],
                                                               self.scaling, self.cosmology))

            # Account for radial dependence for X-ray observables
            if obsnames[i] in ('Mgas', 'Yx'):
                correction = self.conversion_factor_Xray_obs_r500ref(dataID)
                obsArrTemp *= correction
            # Truncate at richness cut
            elif obsnames[i] == 'richness':
                if self.todo['lambda_min']:
                    if self.catalog['FIELD'][dataID] == 'SPTPOL_500d':
                        this_lambda_min = self.lambda_min['deep'](self.catalog['REDSHIFT'][dataID])
                    else:
                        this_lambda_min = self.lambda_min['shallow'](self.catalog['REDSHIFT'][dataID])
                    idx = obsArrTemp > this_lambda_min
                    Delta_x0 = this_lambda_min - obsArrTemp[idx[0]-1]
                    Delta_x = obsArrTemp[idx[0]]-obsArrTemp[idx[0]-1]
                    if i == 0:
                        with np.errstate(invalid='ignore'):
                            Delta_y = lnHMF_3d[idx[0], :, :]-lnHMF_3d[idx[0]-1, :, :]
                            lnHMF_3d_at_cut = lnHMF_3d[idx[0]-1, :, :] + Delta_y*Delta_x0/Delta_x
                        lnHMF_3d = np.insert(lnHMF_3d[idx, :, :], 0, lnHMF_3d_at_cut, axis=0)
                    elif i == 1:
                        with np.errstate(invalid='ignore'):
                            Delta_y = lnHMF_3d[:, idx[0], :]-lnHMF_3d[:, idx[0]-1, :]
                            lnHMF_3d_at_cut = lnHMF_3d[:, idx[0]-1, :] + Delta_y*Delta_x0/Delta_x
                        lnHMF_3d = np.insert(lnHMF_3d[:, idx, :], 0, lnHMF_3d_at_cut, axis=1)
                    obsArrTemp = np.insert(obsArrTemp[idx], 0, this_lambda_min)
            obsArr.append(obsArrTemp)
            lnobsArr.append(np.log(obsArrTemp))

        # SZ arrays
        zeta_arr = np.exp(scaling_relations.lnmass2lnobs('zeta', self.HMF_convos['lnM_arr'], self.catalog['REDSHIFT'][dataID],
                                                         self.scaling, self.cosmology,
                                                         SPTfield=self.SPT_survey[self.SPT_survey['FIELD'] == self.catalog['FIELD'][dataID]]))
        xi_arr = scaling_relations.zeta2xi(zeta_arr)
        if (xi_arr[0] > 2.7) | (self.catalog['XI'][dataID] > xi_arr[-1]-2):
            return 0

        # P(ln(obs0, obs1) | xi)
        dP_dlnobs = self.convolve_HMF_lnobs_to_xi(self.catalog['XI'][dataID], xi_arr, lnHMF_3d)

        # Go to linear space [obs0][obs1]
        dP_dobs01 = dP_dlnobs/obsArr[0][:, None]/obsArr[1][None, :]

        # Normalize
        N = np.trapezoid(np.trapezoid(dP_dobs01, obsArr[1], axis=1), obsArr[0])
        dP_dobs01 /= N

        # P(obs0, obs1)
        # P(Mwl) from data
        WL_interp = InterpolatedUnivariateSpline(np.log(self.WL.M_arr), self.catalog['lnp_Mwl'][dataID], k=1)
        Pwl = np.exp(WL_interp(lnobsArr[0]))
        if obsnames[1] == 'richness':
            idx_lo = np.digitize(self.catalog['richness'][dataID], obsArr[1]) - 1
            if (self.catalog['richness'][dataID] < obsArr[1][0]) | (self.catalog['richness'][dataID] > obsArr[1][-1]):
                print('Problem with richness', self.catalog['richness'][dataID], self.catalog['SPT_ID'][dataID])
                return 0
            Delta_l = (self.catalog['richness'][dataID]-obsArr[1][idx_lo]) / (obsArr[1][idx_lo+1]-obsArr[1][idx_lo])
            # dP_dobs01 can be zero so we're ignoring all warnings here
            with np.errstate(all='ignore'):
                Delta_lny = np.log(dP_dobs01[:, idx_lo+1]/dP_dobs01[:, idx_lo])
                lndP_dobs0 = np.log(dP_dobs01[:, idx_lo]) + Delta_l*Delta_lny
            finite_idx = np.isfinite(lndP_dobs0)
            if np.all(finite_idx is False):
                with np.errstate(divide='ignore'):
                    lndP_dobs01 = np.log(dP_dobs01)
                dP_dobs0 = np.zeros(len(lnobsArr[0]))
                for i in range(len(lnobsArr[0])):
                    if np.any(np.isfinite(lndP_dobs01[i, :])):
                        finite_idx = np.isfinite(lndP_dobs01[i, :])
                        interp = interp1d(lnobsArr[1][finite_idx], lndP_dobs01[i, finite_idx],
                                          kind='linear', fill_value='extrapolate', assume_sorted=True)
                        dP_dobs0[i] = np.exp(interp(np.log(self.catalog['richness'][dataID])))
            else:
                lndP_dobs0_interp = interp1d(lnobsArr[0][finite_idx], lndP_dobs0[finite_idx],
                                             kind='linear', fill_value='extrapolate', assume_sorted=True)
                dP_dobs0 = np.exp(lndP_dobs0_interp(lnobsArr[0]))
            likeli = np.trapezoid(dP_dobs0*Pwl, obsArr[0])

        else:
            Px = norm.pdf(obsmeas[1], obsArr[1], obserr[1])
            Pobs = Pwl[:, None] * Px[None, :]
            likeli = np.trapezoid(np.trapezoid(dP_dobs01*Pobs, obsArr[1], axis=1), obsArr[0])

        if (likeli == 0.) | np.isnan(likeli) | np.isinf(likeli):
            print(self.catalog['SPT_ID'][dataID], obsnames, likeli)
            # np.savetxt(self.catalog['SPT_ID'][dataID], dP_dobs01)
            # np.savetxt(self.catalog['SPT_ID'][dataID]+'1d', (obsArr[0], dP_dobs0, Pwl))

        return likeli
