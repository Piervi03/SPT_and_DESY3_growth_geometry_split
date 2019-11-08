from __future__ import division
import numpy as np
from numpy.lib import scimath as sm
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from multiprocessing import Pool

import h5py
import imp
# import time

import cosmo, Mconversion_concentration, miscentering

########################################
marginalize_conc = True
grid_z_min = .25
grid_z_max = 1
grid_lgM_min = 12.5
grid_lgM_max = 15.5
# Radius [Mpc/h]
len_r_fine = 64
grid_lgr_fine_min = -4
grid_lgr_fine_max = 1

# SPT positional uncertainty
grid_sigma_SPT_min = np.sqrt(1.3**2+.25**2)/50
grid_sigma_SPT_max = np.sqrt(1.3**2+3.1**2)/5

def unwrap_self_precompute_sigmaSPT(arg):
    return SPTlensing.compute_grid_c_sigmaSPT(*arg)

def unwrap_self_precompute_optical(arg):
    return SPTlensing.compute_grid_c(*arg)

def unwrap_self_like_cluster(arg):
    return SPTlensing.like_cluster(*arg)

##### This class reads and stores shear data and calculates P(shear|P(M))
class SPTlensing:

    def __init__(self, catalog, WLsimcalibfile, HSTfile, MegacamFile, DESfile, DES_betabias_file):
        WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
        self.WLcalib = WLsimcalib.WLcalibration
        # beta bias redshift-interpolation for DES
        # beta_true/beta_meas = Sigma_crit,meas/Sigma_crit,true
        data_ = np.loadtxt(DES_betabias_file, unpack=True)[:3]
        self.DES_betabias_mean = InterpolatedUnivariateSpline(data_[1], data_[0])
        self.DES_betabias_var = InterpolatedUnivariateSpline(data_[1], data_[2]**2)
        # Miscentering
        self.DES_miscenterer = miscentering.MisCentering(kind=self.WLcalib['DES_miscenter_kind'])
        self.Delta_crit = self.WLcalib['Delta_crit']

        self.len_c_arr = 8
        self.c_arr = np.logspace(0, np.log10(30), self.len_c_arr)
        self.z_arr = np.logspace(np.log10(grid_z_min), np.log10(grid_z_max), 8)
        self.r_fine = np.logspace(grid_lgr_fine_min, grid_lgr_fine_max, len_r_fine)
        self.r_data_idx = (self.r_fine<.7*.25).nonzero()[0][-1], (self.r_fine>.7*3).nonzero()[0][0]

        self.len_M_arr = 8
        self.M_arr = np.logspace(grid_lgM_min, grid_lgM_max, self.len_M_arr)
        self.sigmaSPT_arr = np.linspace(grid_sigma_SPT_min, grid_sigma_SPT_max, 8)

        self.NPROC = 8

        readdata(catalog, HSTfile, MegacamFile, DESfile)

        # Pre-compute Cholesky decomposition
        WL_idx = ((catalog['WLdata'] != None)&(catalog['REDSHIFT']<1)).nonzero()[0]
        for i in WL_idx:
            catalog[i]['WLdata']['cho_factor'] = cho_factor(catalog[i]['WLdata']['shearcovmat'])


    ########################################
    def like_all(self, catalog, cosmology, scaling, MCrel=None):
        """Return p(data|M_arr) for all clusters with WL data."""
        self.cosmology = cosmology
        self.scaling = scaling
        # self.MCrel = MCrel

        # Pre-compute angular diameter distance and miscentered shear profiles
        self.get_dAs(cosmology)
        # t0 = time.time()
        self.compute_on_grid(cosmology)
        # print "precompute done", time.time()-t0

        # Go through all clusters with WL data
        WL_idx = ((catalog['WLdata'] != None)&(catalog['REDSHIFT']<1)).nonzero()[0]

        if self.NPROC==0:
            p_Mwl = [self.like_cluster(catalog[i], self.M_arr) for i in WL_idx]
        else:
            pool = Pool(processes=self.NPROC)
            argin = zip([self]*len(WL_idx), catalog[WL_idx], [self.M_arr]*len(WL_idx))
            p_Mwl = pool.map(unwrap_self_like_cluster, argin)
            pool.close()

        catalog['p_Mwl'] = [None]*len(catalog)
        catalog['p_Mwl'][WL_idx] = p_Mwl
        #print "lensing done", time.time()-t0


    ########################################
    # Get P(Mwl) from dP/dMwl and shear data
    def like_cluster(self, data, mArr):
        """Return likelihood of shear profile for a given cluster (index) given
        an array of cluster masses."""
        self.cat_cl = data

        ##### Cosmology and halo stuff, in 1/h units
        self.get_beta(self.cosmology)

        ##### Likelihood
        if self.cat_cl['WLdata']['datatype']=='DES':
            pOfMass = self.like_DES_marginalize_conc(mArr, self.cosmology, self.scaling)
        elif self.cat_cl['WLdata']['datatype']=='Megacam':
            rho_c_z, Dl, delta_c, r_s = self.get_cluster_properties(mArr, self.MCrel, self.cosmology)
            pOfMass = self.like_Megacam(rho_c_z, Dl, delta_c, r_s)
        elif self.cat_cl['WLdata']['datatype']=='HST':
            rho_c_z, Dl, delta_c, r_s = self.get_cluster_properties(mArr, self.MCrel, self.cosmology)
            pOfMass = self.like_HST(rho_c_z, Dl, delta_c, r_s)

        return pOfMass



    ########################################


    def likelihood_DES(self, g_t):
        """Return P(DES data|Mwl)"""
        diff_ = self.cat_cl['WLdata']['shear'] - g_t
        chi2 = np.dot(diff_, cho_solve(self.cat_cl['WLdata']['cho_factor'], diff_))
        likeli = np.exp(-.5 * chi2)

        return likeli




    def like_DES_marginalize_conc(self, mArr, cosmology, scaling):
        """Return array P(DES data|Mwl), marginalized over concentration in the
        range 0<c<30. Note that this is not normalized wrt the mArr for a good
        reason: In general, the mArr will not cover the full pOfMass range, and
        it varies as a function of SZ parameters. However, pOfMass is a product
        of normalized distributions, and so its normalization is constant
        throughout parameter space."""
        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
        Dl = cosmo.dA(self.cat_cl['REDSHIFT'], cosmology)
        Sigma_c = 1.6624541593797974e+18/Dl/self.beta_avg
        r_arcmin = self.r_fine[self.r_data_idx[0]:self.r_data_idx[1]] / Dl * 60*180/np.pi

        # Interpolate to z
        idx_z_lo = (self.z_arr<=self.cat_cl['REDSHIFT']).nonzero()[0][-1]
        Delta_lnz = np.log(self.cat_cl['REDSHIFT'] / self.z_arr[idx_z_lo])


        if self.DES_miscenterer.kind in ['r200', 'arcmin']:
            # [c, M, draw]
            y_lo = self.Sigma_weights[:,idx_z_lo,:,:]
            weights = y_lo + Delta_lnz * (self.Sigma_weights[:,idx_z_lo+1,:,:]-y_lo)
            # [c, M, r, draw]
            y_lo = self.lnSigma_draws[:,idx_z_lo,:,:,:]
            lnSigma_draws = y_lo + Delta_lnz * (self.lnSigma_draws[:,idx_z_lo+1,:,:,:]-y_lo)
            y_lo = self.lnDelta_Sigma_draws[:,idx_z_lo,:,:,:]
            lnDelta_Sigma_draws = y_lo + Delta_lnz * (self.lnDelta_Sigma_draws[:,idx_z_lo+1,:,:,:]-y_lo)

        elif self.DES_miscenterer.kind=='SPT':
            # Interpolate to sigma_SPT
            this_sigmaSPT = np.sqrt(1.3**2 + self.cat_cl['THETA_CORE']**2)/self.cat_cl['XI']
            idx_sig_lo = (self.sigmaSPT_arr<=this_sigmaSPT).nonzero()[0][-1]
            Delta_sig = this_sigmaSPT - self.sigmaSPT_arr[idx_sig_lo]
            # [c, M, draw]
            y_lo = self.Sigma_weights[:,idx_sig_lo,idx_z_lo,:,:]
            weights = y_lo + Delta_sig * (self.Sigma_weights[:,idx_sig_lo+1,idx_z_lo,:,:]-y_lo) \
                                + Delta_lnz * (self.Sigma_weights[:,idx_sig_lo,idx_z_lo+1,:,:]-y_lo)
            # [c, M, r, draw]
            y_lo = self.lnSigma_draws[:,idx_sig_lo,idx_z_lo,:,:,:]
            lnSigma_draws = y_lo + Delta_sig * (self.lnSigma_draws[:,idx_sig_lo+1,idx_z_lo,:,:,:]-y_lo) \
                                      + Delta_lnz * (self.lnSigma_draws[:,idx_sig_lo,idx_z_lo+1,:,:,:]-y_lo)
            y_lo = self.lnDelta_Sigma_draws[:,idx_sig_lo,idx_z_lo,:,:,:]
            lnDelta_Sigma_draws = y_lo + Delta_sig * (self.lnDelta_Sigma_draws[:,idx_sig_lo+1,idx_z_lo,:,:,:]-y_lo) \
                                      + Delta_lnz * (self.lnDelta_Sigma_draws[:,idx_sig_lo,idx_z_lo+1,:,:,:]-y_lo)

        ##### Apply mean beta bias
        betabias_mean_ = self.DES_betabias_mean(self.cat_cl['REDSHIFT'])
        betabias_var_ = self.DES_betabias_var(self.cat_cl['REDSHIFT'])
        Sigma_c/= betabias_mean_

        ##### Reduced shear profile [c,M,r,draw]
        gamma = np.exp(lnDelta_Sigma_draws)/Sigma_c
        kappa = np.exp(lnSigma_draws)/Sigma_c
        g_t_draws = gamma / (1 - kappa)

        # Error on shear due to error on Sigma
        rel_var_Sigmac = betabias_var_/Sigma_c**2
        rel_varshear_varbeta = rel_var_Sigmac + kappa**2*rel_var_Sigmac/(1-kappa)**2 - 2*kappa/(1-kappa)*rel_var_Sigmac

        # Realization of shear and beta bias
        total_std_ = g_t_draws * np.sqrt(rel_varshear_varbeta + self.WLcalib['DESshearErr']**2 + self.WLcalib['DEScontamCorr']**2)
        g_t_draws+= scaling['DESbias'] * total_std_

        del lnSigma_draws
        del lnDelta_Sigma_draws
        del gamma
        del kappa


        p_M_c_draw = np.empty((self.len_M_arr, self.len_c_arr, self.DES_miscenterer.len_Rmis))
        for i in range(self.len_M_arr):
            for j in range(self.len_c_arr):
                for k in range(self.DES_miscenterer.len_Rmis):
                    g_t_interp = InterpolatedUnivariateSpline(r_arcmin, g_t_draws[j,i,:,k])
                    this_g_t = g_t_interp(self.cat_cl['WLdata']['r_arcmin'])
                    p_M_c_draw[i,j,k] = self.likelihood_DES(this_g_t) * weights[j,i,k]
        p_M_c = np.sum(p_M_c_draw, axis=-1)
        likeli = np.trapz(p_M_c, self.c_arr, axis=1)

        return likeli


    ########################################

    def compute_grid_c(self, i):
        """Pre-compute shear profile grid [sigma_SPT, z, M, r] for a given
        concentration `c`."""
        c = self.c_arr[i]
        delta_c = self.Delta_crit/3 * c**3 / (np.log(1+c) - c/(1+c))
        # [z, M]
        r_Delta = (3*self.M_arr/4/np.pi/self.Delta_crit/self.rho_c_z[:,None])**(1/3)
        r_s = r_Delta/c
        # [z, M, r]
        x = self.r_fine[None,None,:] / r_s[:,:,None]
        Sigma = get_Sigma(x, r_s[:,:,None], self.rho_c_z[:,None,None], delta_c)
        # [z, r]
        r_arcmin = self.r_fine[None,:] / self.Dl_arr[:,None] * 60*180/np.pi

        lnSigma_draws = np.empty((len(self.z_arr),self.len_M_arr,self.r_data_idx[1]-self.r_data_idx[0],self.DES_miscenterer.len_Rmis))
        Sigma_weights = np.empty((len(self.z_arr),self.len_M_arr,self.DES_miscenterer.len_Rmis))
        lnDelta_Sigma_draws = np.empty((len(self.z_arr),self.len_M_arr,self.r_data_idx[1]-self.r_data_idx[0],self.DES_miscenterer.len_Rmis))

        for j,z in enumerate(self.z_arr):
            for k,m in enumerate(self.M_arr):
                draws_, Rmis_, p_Rmis_ = self.DES_miscenterer.get_profile_mean_draws(r_arcmin[j,:], Sigma[j,k],
                                                                                     4*np.amax((self.DES_miscenterer.sigma0, self.DES_miscenterer.sigma1)),
                                                                                      # r200c=r_Delta[j,k],
                                                                                     )

                # Weights
                Sigma_weights[j,k] = p_Rmis_ * Rmis_[-1]/self.DES_miscenterer.len_Rmis

                # Sigma (surface mass density) [r, draws]
                draws_[draws_<=0] = np.nextafter(0,1)
                lnSigma_draws[j,k] = np.log(draws_[self.r_data_idx[0]:self.r_data_idx[1],:])

                # Sigma_bar (avg. density inside radius r)
                integrands = [InterpolatedUnivariateSpline(self.r_fine, self.r_fine*draws_[:,l])
                              for l in range(draws_.shape[1])]
                mean_Sigma = np.array([2/r**2 * integrands[l].integral(self.r_fine[0], r)
                                        for r in self.r_fine for l in range(draws_.shape[1])]).reshape(len(self.r_fine),-1)

                # Delta Sigma
                Delta_Sigma = mean_Sigma - draws_
                Delta_Sigma[Delta_Sigma<=0] = np.nextafter(0,1)
                lnDelta_Sigma_draws[j,k] = np.log(Delta_Sigma[self.r_data_idx[0]:self.r_data_idx[1],:])

        return lnSigma_draws, lnDelta_Sigma_draws, Sigma_weights





    ########################################

    def compute_grid_c_sigmaSPT(self, i):
        """Pre-compute shear profile grid [sigma_SPT, z, M, r] for a given
        concentration `c`."""
        c = self.c_arr[i]
        delta_c = self.Delta_crit/3 * c**3 / (np.log(1+c) - c/(1+c))
        # MCrel = Mconversion_concentration.ConcentrationConversion(c, setup_interp=True)
        # [z, M]
        r_Delta = (3*self.M_arr/4/np.pi/self.Delta_crit/self.rho_c_z[:,None])**(1/3)
        r_s = r_Delta/c
        # [z, M, r]
        x = self.r_fine[None,None,:] / r_s[:,:,None]
        Sigma = get_Sigma(x, r_s[:,:,None], self.rho_c_z[:,None,None], delta_c)
        # [z, r]
        r_arcmin = self.r_fine[None,:] / self.Dl_arr[:,None] * 60*180/np.pi

        lnSigma_draws = np.empty((len(self.sigmaSPT_arr),len(self.z_arr),self.len_M_arr,self.r_data_idx[1]-self.r_data_idx[0],self.DES_miscenterer.len_Rmis-1))
        Sigma_weights = np.empty((len(self.sigmaSPT_arr),len(self.z_arr),self.len_M_arr,self.DES_miscenterer.len_Rmis-1))
        lnDelta_Sigma_draws = np.empty((len(self.sigmaSPT_arr),len(self.z_arr),self.len_M_arr,self.r_data_idx[1]-self.r_data_idx[0],self.DES_miscenterer.len_Rmis-1))

        for h,sigma_SPT in enumerate(self.sigmaSPT_arr):
            for j,z in enumerate(self.z_arr):
                for k,m in enumerate(self.M_arr):
                    mean_, draws_, weights_ = self.DES_miscenterer.get_profile_mean_draws(r_arcmin[j], Sigma[j,k], .6*self.r500_deg[j,k],
                                                                                          r500_arcmin=self.r500_arcmin[j,k],
                                                                                          sigma_SPT=sigma_SPT/60
                                                                                          )

                    # Weights
                    Sigma_weights[h,j,k] = weights_

                    # Sigma (surface mass density) [r, draws]
                    draws_[draws_<=0] = np.nextafter(0,1)
                    lnSigma_draws[h,j,k] = np.log(draws_[self.r_data_idx[0]:self.r_data_idx[1],:])

                    # Sigma_bar (avg. density inside radius r)
                    integrands = [InterpolatedUnivariateSpline(self.r_fine, self.r_fine*draws_[:,l])
                                  for l in range(draws_.shape[1])]
                    mean_Sigma = np.array([2/r**2 * integrands[l].integral(self.r_fine[0], r)
                                            for r in self.r_fine for l in range(draws_.shape[1])]).reshape(len(self.r_fine),-1)

                    # Delta Sigma
                    Delta_Sigma = mean_Sigma - draws_
                    Delta_Sigma[Delta_Sigma<=0] = np.nextafter(0,1)
                    lnDelta_Sigma_draws[h,j,k] = np.log(Delta_Sigma[self.r_data_idx[0]:self.r_data_idx[1],:])

        return lnSigma_draws, lnDelta_Sigma_draws, Sigma_weights



    ########################################

    def compute_on_grid(self, cosmology):
        """Pre-compute shear profiles for grid [c, z, M, r]"""
        # z-dependent
        self.Dl_arr = np.array([cosmo.dA(z, cosmology) for z in self.z_arr])
        self.rho_c_z = cosmo.RHOCRIT * np.array([cosmo.Ez(z, cosmology)**2  for z in self.z_arr]) # [h^2 Msun/Mpc^3]

        if self.DES_miscenterer.kind=='SPT':
            # [z, M]
            r500c = (3*self.M_arr[None,:]/4/np.pi/500/self.rho_c_z[:,None])**(1/3)
            self.r500_arcmin = r500c / self.Dl_arr[:,None] * 60*180/np.pi

        if self.NPROC==0:
            if self.DES_miscenterer.kind in ['r200', 'arcmin']:
                out = [self.compute_grid_c(i) for i in range(self.len_c_arr)]
            elif self.DES_miscenterer.kind=='SPT':
                out = [self.compute_grid_c_sigmaSPT(i) for i in range(self.len_c_arr)]
        else:
            pool = Pool(processes=self.NPROC)
            argin = zip([self]*len(self.c_arr), range(self.len_c_arr))
            if self.DES_miscenterer.kind in ['r200', 'arcmin']:
                out = pool.map(unwrap_self_precompute_optical, argin)
            elif self.DES_miscenterer.kind=='SPT':
                out = pool.map(unwrap_self_precompute_sigmaSPT, argin)
            pool.close()

        self.lnSigma_draws = np.array([out[i][0] for i in range(self.len_c_arr)])
        self.Sigma_weights = np.array([out[i][2] for i in range(self.len_c_arr)])
        self.lnDelta_Sigma_draws = np.array([out[i][1] for i in range(self.len_c_arr)])


    ########################################

    def get_cluster_properties(self, mArr, MCrel, cosmology):
        """Return rho_c(z_cluster), luminosity distance (z_cluster), delta_c,
        and r_s."""
        Dl = cosmo.dA(self.cat_cl['REDSHIFT'], cosmology)
        rho_c_z = cosmo.RHOCRIT * cosmo.Ez(self.cat_cl['REDSHIFT'], cosmology)**2 # [h^2 Msun/Mpc^3]
        if marginalize_conc:
            M200c = np.exp(MCrel.lnM_to_lnM200(np.log(mArr)))
        else:
            M200c = np.exp(MCrel.lnM_to_lnM200(self.cat_cl['REDSHIFT'], np.log(mArr)))[0]
        r200c = (3*M200c/4/np.pi/200/rho_c_z)**(1/3)
        c200c = MCrel.calC200(M200c, self.cat_cl['REDSHIFT'])
        delta_c = 200/3 * c200c**3 / (np.log(1+c200c) - c200c/(1+c200c))
        r_s = r200c/c200c
        return rho_c_z, Dl, delta_c, r_s


    ########################################

    def like_Megacam(self, rho_c_z, Dl, delta_c, r_s):
        """Return array P(Megacam data|Mwl). Note that this is not normalized
        wrt the mArr for a good reason: In general, the mArr will not cover the
        full pOfMass range, and it varies as a function of SZ parameters.
        However, pOfMass is a product of normalized distributions, and so its
        normalization is constant throughout parameter space."""
        # Dimensionless radial distance [Radius][Mass]
        x = self.cat_cl['WLdata']['r_deg'][:,None] * Dl * np.pi/180 / r_s[None,:]
        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
        Sigma_c = 1.6624541593797974e+18/Dl/self.beta_avg

        # gamma_t, kappa, g_t [Radius][Mass]
        gamma_2d = get_Delta_Sigma(x, r_s, rho_c_z, delta_c) / Sigma_c
        kappa_2d = get_Sigma(x, r_s, rho_c_z, delta_c) / Sigma_c
        g_2d = gamma_2d/(1-kappa_2d) * (1 + kappa_2d*(self.beta2_avg/self.beta_avg**2-1))

        likelihood = norm.pdf(g_2d, self.cat_cl['WLdata']['shear'][:,None], self.cat_cl['WLdata']['shearerr'][:,None])

        pOfMass = np.prod(likelihood, axis=0)

        return pOfMass



    ########################################

    def like_HST(self, rho_c_z, Dl, delta_c, r_s):
        """Return array P(HST data|Mwl). Note that this is not normalized wrt
        the mArr for a good reason: In general, the mArr will not cover the full
        pOfMass range, and it varies as a function of SZ parameters. However,
        pOfMass is a product of normalized distributions, and so its
        normalization is constant throughout parameter space."""
        # Dimensionless radial distance [Radius][Mass]
        x = self.cat_cl['WLdata']['r_deg'][:,None] * Dl * np.pi/180 / r_s[None,:]

        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2] [Radius]
        rangeR = range(len(self.cat_cl['WLdata']['r_deg']))
        betaR = np.array([self.beta_avg[self.cat_cl['WLdata']['magbinids'][i]] for i in rangeR])
        beta2R = np.array([self.beta2_avg[self.cat_cl['WLdata']['magbinids'][i]] for i in rangeR])
        Sigma_c = 1.6624541593797974e+18/Dl/betaR

        # gamma_t and kappa [Radius][Mass]
        gamma_2d = get_Delta_Sigma(x, r_s, rho_c_z, delta_c) / Sigma_c[:,None]
        kappa_2d = get_Sigma(x, r_s, rho_c_z, delta_c) / Sigma_c[:,None]

        # [Radius][Mass]
        mu0_2d = 1/((1-kappa_2d)**2 - gamma_2d**2)
        kappaFake = (mu0_2d-1)/2

        # Magnification correction [Radius][Mass]
        mykappa = kappaFake * 0.3/betaR[:,None]

        magcorr = [np.interp(mykappa[i], self.cat_cl['WLdata']['magcorr'][self.cat_cl['WLdata']['magbinids'][i]][0], self.cat_cl['WLdata']['magcorr'][self.cat_cl['WLdata']['magbinids'][i]][1]) for i in rangeR]

        # Beta correction [Radius][Mass]
        betaratio = beta2R/betaR**2
        betaCorr = (1 + kappa_2d*(betaratio[:,None]-1))

        # Reduced shear g_t [Radius][Mass]
        g_2d = np.array(magcorr) * gamma_2d/(1-kappa_2d) * betaCorr

        # Only consider 500<r/kpc/1500 in reference cosmology
        cosmoRef = {'Omega_m':.3, 'Omega_l':.7, 'h':.7, 'w0':-1., 'wa':0}
        DlRef = cosmo.dA(self.cat_cl['REDSHIFT'], cosmoRef)
        rPhysRef = self.cat_cl['WLdata']['r_deg'] * DlRef * np.pi/180 /cosmoRef['h']
        rInclude = np.where((rPhysRef>.5)&(rPhysRef<1.5))[0]

        likelihood = norm.pdf(g_2d[rInclude,:], self.cat_cl['WLdata']['shear'][rInclude,None], self.cat_cl['WLdata']['shearerr'][rInclude,None])

        pOfMass = np.prod(likelihood, axis=0)

        return pOfMass




    ########################################
    # dA [Mpc/h]
    def get_dAs(self, cosmology):
        """Precompute angular diameter distances for an array of redshifts."""
        zs = np.logspace(-1,np.log10(5),100)
        dA = np.array([cosmo.dA(z, cosmology) for z in zs])
        self.dAs = {'lnz':np.log(zs), 'lndA':np.log(dA)}


    ########################################
    def get_beta(self, cosmology):
        """Compute <beta> and <beta^2> from distribution of redshift galaxies."""
        ##### Only consider redshift bins behind the cluster
        betaArr = np.zeros(len(self.cat_cl['WLdata']['redshifts']))
        bgIdx = np.where(self.cat_cl['WLdata']['redshifts']>self.cat_cl['REDSHIFT'])[0]

        ##### Calculate beta(z_source)
        # Set up interpolation
        z_arr = np.linspace(np.amin(self.cat_cl['WLdata']['redshifts'][bgIdx]), np.amax(self.cat_cl['WLdata']['redshifts'][bgIdx]), 64)
        dA_ls = np.array([cosmo.dA_two_z(self.cat_cl['REDSHIFT'], z, cosmology) for z in z_arr])
        dA_ls_interp = InterpolatedUnivariateSpline(z_arr, dA_ls)
        # beta = dA_ls / dA_l
        betaArr[bgIdx] = dA_ls_interp(self.cat_cl['WLdata']['redshifts'][bgIdx])
        betaArr[bgIdx]/= np.exp(np.interp(np.log(self.cat_cl['WLdata']['redshifts'][bgIdx]), self.dAs['lnz'], self.dAs['lndA']))

        ##### Weight beta(z) with N(z) distribution to get <beta> and <beta^2>
        if self.cat_cl['WLdata']['datatype']=='Megacam':
            self.beta_avg = np.sum(self.cat_cl['WLdata']['Nz']*betaArr)/self.cat_cl['WLdata']['Ntot']
            self.beta2_avg = np.sum(self.cat_cl['WLdata']['Nz']*betaArr**2)/self.cat_cl['WLdata']['Ntot']
        elif self.cat_cl['WLdata']['datatype']=='DES':
            self.beta_avg = np.mean(betaArr)
            self.beta2_avg = np.mean(betaArr**2)
        else:
            self.beta_avg, self.beta2_avg = {}, {}
            for i in self.cat_cl['WLdata']['pzs'].keys():
                self.beta_avg[i] = np.sum(self.cat_cl['WLdata']['pzs'][i]*betaArr)/self.cat_cl['WLdata']['Ntot'][i]
                self.beta2_avg[i] = np.sum(self.cat_cl['WLdata']['pzs'][i]*betaArr**2)/self.cat_cl['WLdata']['Ntot'][i]


################################################################################
# NFW tools by Joerg Dietrich (https://github.com/joergdietrich/NFW)

def arcsec(z):
    """Compute the inverse sec of the complex number z."""
    val1 = 1j / z
    val2 = sm.sqrt(1 - 1/z**2)
    val = 1j * np.log(val2 + val1)
    return .5 * np.pi + val

def get_Delta_Sigma(x, r_s, rho_c_z, delta_c):
    """Return Delta Sigma"""
    fac = 2 * r_s * rho_c_z * delta_c
    val1 = 1 / (1 - x**2)
    num = ((3 * x**2) - 2) * arcsec(x)
    div = x**2 * (sm.sqrt(x**2 - 1))**3
    val2 = (num / div).real
    val3 = 2 * np.log(x / 2) / x**2
    return fac * (val1+val2+val3)

def get_Sigma(x, r_s, rho_c_z, delta_c):
    """Return Sigma_NFW"""
    fac = 2 * r_s * rho_c_z * delta_c
    val1 = 1 / (x**2 - 1)
    val2 = (arcsec(x) / (sm.sqrt(x**2 - 1))**3).real
    return fac * (val1-val2)



################################################################################
def readdata(catalog, HSTfile, MegacamFile, DESfile):
    """Read and load weak-lensing data into `WLdata` field in `catalog` if
    the corresponding path-variables lead to valid files on disk."""
    # Empty WL data field
    catalog['WLdata'] = [None for i in range(len(catalog['SPT_ID']))]

    ##### Check for HST data
    if HSTfile != 'None':
        with h5py.File(HSTfile, 'r') as f:
            for i,name in enumerate(catalog['SPT_ID']):
                if name in f.keys():
                    catalog['WLdata'][i] = {'datatype':'HST', 'center':f[name].attrs['center'],
                        'r_deg':f[name]['shear_profile'][0], 'shear':f[name]['shear_profile'][1], 'shearerr':f[name]['shear_profile'][2], 'magbinids':f[name]['shear_profile'][3],
                        'redshifts':f[name]['redshifts'],
                        'pzs':{}, 'magcorr':{}, 'Ntot':{}}
                    for key in f[name]['magbindata'].keys():
                        catalog['WLdata'][i]['pzs'][key] = np.sum(f[name]['magbindata'][key]['pzs'], axis=0)
                        catalog['WLdata'][i]['Ntot'][key] = np.sum(catalog['WLdata'][i]['pzs'][key])
                        catalog['WLdata'][i]['magcorr'][key] = f[name]['magbindata'][key]['magnificationcorr']

    ##### Megacam data
    if MegacamFile != 'None':
        with h5py.File(MegacamFile, 'r') as f:
            for i,name in enumerate(catalog['SPT_ID']):
                if name in f.keys():
                    catalog['WLdata'][i] = {'datatype':'Megacam',
                        'r_deg':f[name]['shear_profile'][0], 'shear':f[name]['shear_profile'][1], 'shearerr':f[name]['shear_profile'][2],
                        'redshifts':f[name]['Nz'][0], 'Nz':f[name]['Nz'][1], 'Ntot':np.sum(f[name]['Nz'][1]),}

    ##### Check for DES data
    if DESfile != 'None':
        with h5py.File(DESfile, 'r') as f:
            for i,name in enumerate(catalog['SPT_ID']):
                if name in f.keys():
                    catalog['WLdata'][i] = {'datatype':'DES',
                        'r_arcmin': f[name]['shear_profile'][0],
                        'shear': f[name]['shear_profile'][1],
                        'shearcovmat': f[name]['shear_profile_cov'][:],
                        'redshifts': f[name]['Nz'][:],
                        'R_mis_arcmin': f[name]['R_mis_arcmin'][()],}
