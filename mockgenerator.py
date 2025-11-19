import numpy as np
import sys
import time
import importlib
from scipy.interpolate import make_interp_spline
from astropy.table import Table

import cosmo
import scaling_relations

# Reference cosmology for which Mgas is measured
cosmologyRef = {'Omega_m': .272, 'Omega_l': .728, 'h': .702, 'w0': -1, 'wa': 0}


def main(configMod_file, catalog_name):
    # Input parameters and settings
    spec = importlib.util.spec_from_file_location('dummy', configMod_file)
    configMod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configMod)
    # SPT survey information
    SPT_survey = Table.read(configMod.SPT_survey, format='ascii.commented_header')
    # lambda_min(z)
    tmp = np.genfromtxt(configMod.MCMF_lambda_min, names=True, dtype=None)
    surveyCutLambda = {}
    for name in tmp.dtype.names[1:]:
        surveyCutLambda[name] = make_interp_spline(tmp['z'], tmp[name], k=1)

    cosmology = configMod.cosmology
    scaling = configMod.scaling
    rng = np.random.default_rng(configMod.random_seed)

    # Set up HMF
    M_arr = np.logspace(13, 16, 301)
    lnM_arr = np.log(M_arr)
    dlnm = np.log(M_arr[1]/M_arr[0])
    dz_ = .01
    z_arr = np.linspace(configMod.z_cl_min_max[0], configMod.z_cl_min_max[1], int((configMod.z_cl_min_max[1]-configMod.z_cl_min_max[0])/dz_) + 1)
    dz = z_arr[1]-z_arr[0]
    if configMod.HMF == 'Mira-Titan':
        import compute_HMF_MiraTitan
        MiraTitan_HMF = compute_HMF_MiraTitan.HMFCalculator(200., 'Duffy08', z_arr, M_arr)
        bad = MiraTitan_HMF.compute_HMF(cosmology)
        if bad:
            print("Could not compute mass function")
        dNdlnM = MiraTitan_HMF.dNdlnM
    else:
        import baccoemu
        import compute_HMF_Bocquet16
        import compute_HMF_Tinker08
        emulator = baccoemu.Matter_powerspectrum()
        params = {'A_s': 1e-10*np.exp(cosmology['ln1e10As'])}
        if 'mnu' not in cosmology.keys():
            cosmology['mnu'] = cosmology['Omnuh2'] * 94.06410581217612 / (cosmology['nnu']/3.)**.75 / (2.7255/2.7255)**3
        else:
            cosmology['Omnuh2'] = cosmology['mnu'] * (cosmology['nnu']/3.)**.75 * (2.7255/2.7255)**3 / 94.06410581217612
        cosmology['Omega_nu'] = cosmology['Omnuh2'] / cosmology['h']**2
        for me, bacco in zip(['Omega_m', 'Omega_b', 'mnu', 'h', 'n_s', 'w0', 'wa'],
                             ['omega_matter', 'omega_baryon', 'neutrino_mass', 'hubble', 'ns', 'w0', 'wa']):
            params[bacco] = cosmology[me]
        # Call the emulator for P_{CDM+bar}(k)
        k, Pk = emulator.get_linear_pk(expfactor=1./(1.+z_arr),
                                       cold=True,
                                       **params)
        # Compute sigma_8 for total matter
        k_, Pk_ = emulator.get_linear_pk(expfactor=1.,
                                         cold=False,
                                         **params)
        kR = 8.*k_
        window = 3. * (np.sin(kR)/kR**3 - np.cos(kR)/kR**2)
        integrand_sigma2 = Pk_ * window**2 * k_**3
        sigma8_squ = .5/np.pi**2 * np.trapezoid(integrand_sigma2, np.log(k_))
        print('sigma_8 %.5f' % np.sqrt(sigma8_squ))
        # Initialize fitting function
        if configMod.HMF == 'Tinker08':
            HMF_calculator = compute_HMF_Tinker08.HMFCalculator(200., z_arr, M_arr)
        elif configMod.HMF == 'Bocquet16':
            HMF_calculator = compute_HMF_Bocquet16.HMFCalculator(200., z_arr, M_arr)
        # Compute the mass function and volume element
        dNdlnM_noVol, dNdlnM = HMF_calculator.compute_HMF(cosmology, z_arr, k, Pk)
    HMF_dNdM_V = dNdlnM * dz*dlnm*(np.pi/180)**2
    # First and last redshift bins are only half in the sample
    HMF_dNdM_V[0, :] *= .5
    HMF_dNdM_V[-1, :] *= .5

    # [DES WL, X-ray, SZ, richness, HST WL]
    covs = np.empty((len(z_arr), len(M_arr), 6, 6))
    for i, z in enumerate(z_arr):
        for j, M in enumerate(M_arr):
            scaling['DWL_DES'] = scaling_relations.WLscatter('WLDES', np.log(M), z, scaling)
            scaling['DWL_Euclid'] = scaling_relations.WLscatter('WLEuclid', np.log(M), z, scaling)
            covs[i, j, :, :] = [[scaling['DWL_DES']**2, scaling['rhoWLX']*scaling['DWL_DES']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES'], scaling['rhoWLrichness']*scaling['DWL_DES']*scaling['Drichness'], 0, 0],
                                [scaling['rhoWLX']*scaling['DWL_DES']*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['rhoXrichness']*scaling['Dx']*scaling['Drichness'], scaling['rhoWLX']*scaling['DWL_HST']*scaling['Dx'], scaling['rhoWLX']*scaling['DWL_Euclid']*scaling['Dx']],
                                [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES'], scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2, scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_HST'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Euclid']],
                                [scaling['rhoWLrichness']*scaling['DWL_DES']*scaling['Drichness'], scaling['rhoXrichness']*scaling['Dx']*scaling['Drichness'], scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness'], scaling['Drichness']**2, scaling['rhoWLrichness']*scaling['DWL_HST']*scaling['Drichness'], scaling['rhoWLrichness']*scaling['DWL_Euclid']*scaling['Drichness']],
                                [0, scaling['rhoWLX']*scaling['DWL_HST']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_HST'], scaling['rhoWLrichness']*scaling['DWL_HST']*scaling['Drichness'], scaling['DWL_HST']**2, 0],
                                [0, scaling['rhoWLX']*scaling['DWL_Euclid']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Euclid'], scaling['rhoWLrichness']*scaling['DWL_Euclid']*scaling['Drichness'], 0, scaling['DWL_Euclid']**2]]

    # Get the mock catalog
    # The HMF is in units [Msun/h]
    mock, fieldnames = [], []
    # xiArrEdge = np.linspace(5,10,101)
    # xiArrBin = (xiArrEdge[1:]+xiArrEdge[:-1])/2
    # dxi = xiArrEdge[1]-xiArrEdge[0]

    for fieldidx, field in enumerate(SPT_survey['FIELD']):
        print(field, fieldidx, 'out of %d' % len(SPT_survey['FIELD']))
        # Poisson realization
        N = rng.poisson(HMF_dNdM_V*SPT_survey['AREA'][fieldidx])

        obs_0 = np.array([np.exp(scaling_relations.lnmass2lnobs(name, lnM_arr[None, :], z_arr[:, None], scaling, cosmology, SPTfield=SPT_survey[fieldidx]))
                          for name in ('WLDES', configMod.Xray_obs, 'zeta', 'richness_base', 'WLEuclid')])
        # Hack for HST
        obs_0 = np.insert(obs_0, -1, np.full(obs_0[0].shape, np.exp(lnM_arr))[None, ...], axis=0)

        for i, z in enumerate(z_arr):
            if SPT_survey['LAMBDA_MIN'][fieldidx] not in ['NONE', 'None', 'none']:
                lambda_min = surveyCutLambda[SPT_survey['LAMBDA_MIN'][fieldidx]](z)
            else:
                lambda_min = -1.

            for j, M in enumerate(M_arr):
                if N[i, j] == 0:
                    continue

                obs = np.exp(rng.multivariate_normal(np.log(obs_0[:, i, j]), covs[i, j], N[i, j]))

                keep = (obs[:, 2] > scaling['zeta_min'])
                for k in keep.nonzero()[0]:
                    # draw xi|zeta
                    xi = rng.normal(scaling_relations.zeta2xi(obs[k, 2]), scale=1.)
                    if xi >= SPT_survey['XI_MIN'][fieldidx]:
                        # Apply observational error to X-ray
                        X = rng.lognormal(np.log(obs[k, 1]), sigma=configMod.Xerr)

                        # Observed richness
                        richness_err = 0.
                        if lambda_min == -1.:
                            # No measurement here
                            richness_obs = 0.
                        else:
                            if configMod.richness_scatter_model == 'lognormal':
                                richness_obs = obs[k, 3]
                            elif configMod.richness_scatter_model == 'lognormalrelPoisson':
                                richness_obs = np.exp(rng.normal(np.log(obs[k, 3]), scale=1/np.sqrt(obs[k, 3])))
                            elif configMod.richness_scatter_model in ['lognormalGaussPoisson', 'lognormalGausssuperPoisson']:
                                if configMod.richness_scatter_model == 'lognormalGaussPoisson':
                                    richness_err = np.sqrt(obs[k, 3])
                                elif configMod.richness_scatter_model == 'lognormalGausssuperPoisson':
                                    richness_err = np.sqrt(obs[k, 3]+10.) * (1.08 + .45*(z-.6))
                                richness_obs = rng.normal(obs[k, 3], scale=richness_err)
                            else:
                                raise ValueError("Unknown value for richness_scatter_model")
                            # Cut in richness
                            if richness_obs < lambda_min:
                                continue

                        mock.append([M, z,
                                     obs[k, 2], xi,
                                     X,
                                     obs[k, 0], obs[k, 4], obs[k, 5],
                                     obs[k, 3], richness_obs, richness_err,
                                     SPT_survey['GAMMA'][fieldidx]])
                        fieldnames.append(field)

        # False detections
        # dNdxiFalse = SPT_survey['BETA'][fieldidx] * SPT_survey['AREA'][fieldidx]/2500. * SPT_survey['ALPHA'][fieldidx] * np.exp(-SPT_survey['BETA'][fieldidx]*(xiArrBin-5.)) * dxi
        # for i in range(len(dNdxiFalse)):
        #     N = rng.poisson(dNdxiFalse[i])
        #     for k in range(N):
        #         mock.append((0., 0., xiArrBin[i], 0., 0., 0.))
        #         fieldnames.append(field)

    nCluster = len(mock)
    mock = Table(rows=mock, names=['M_true', 'REDSHIFT',
                                   'zeta', 'XI',
                                   configMod.Xray_obs,
                                   'Mwl_DES_200', 'Mwl_HST_200', 'Mwl_Euclid_200',
                                   'richness_int', 'richness', 'richness_err',
                                   'GAMMA_FIELD'])
    mock['SPT_ID'] = ['cluster%d' % i for i in range(nCluster)]
    mock['FIELD'] = fieldnames
    mock['Mwl_DES_200_obs'] = rng.lognormal(np.log(mock['Mwl_DES_200']), sigma=.8)

    # HST weak lensing
    HST_z_range = ((mock['REDSHIFT'] > .6) & (mock['REDSHIFT'] < 1.1)).nonzero()[0]
    HST_idx = np.argsort(mock['XI'][HST_z_range])[-30:]
    # HST_idx = rng.choice(HST_z_range, 30, replace=False)
    mask = np.ones(len(mock), bool)
    mask[HST_z_range[HST_idx]] = 0
    mock['Mwl_HST_200'][mask] = 0.

    # Select XVP
    print(nCluster, 'clusters', 'xi max', np.amax(mock['XI']), 'lambda min', np.amin(mock['richness']))

    # Select nXrayCluster highest xi for Yx follow-up
    mock['ln%s_err' % configMod.Xray_obs] = np.full(nCluster, configMod.Xerr)
    XVP = np.argsort(mock['XI'])
    mock[configMod.Xray_obs][XVP[:-configMod.nXrayCluster]] = 0.
    mock['ln%s_err' % configMod.Xray_obs][XVP[:-configMod.nXrayCluster]] = 0.

    # Create X-ray gas mass profiles
    # For maximal confusion, this part is in decent units, with factors of h
    # because Xrayprofile.py is in nice units as well :)
    mock['%s_profile' % configMod.Xray_obs] = np.zeros((nCluster, 2, 80))
    r_ref = np.linspace(25, 2000, 80)
    for i in range(nCluster):
        if mock[configMod.Xray_obs][i] == 0.:
            continue
        # Angular diameter distances
        # The reference cosmology matches Mike M's choice for the XVP data
        dAref = cosmo.dA(mock['REDSHIFT'][i], cosmologyRef) / cosmologyRef['h']
        dA = cosmo.dA(mock['REDSHIFT'][i], cosmology) / cosmology['h']
        # Scale r_ref to current cosmo
        rArr = r_ref * dA/dAref
        # Get the true r500
        rho_c_z = cosmo.RHOCRIT * cosmo.Ez(mock['REDSHIFT'][i], cosmology)**2.
        r500 = 1000 * (3*mock['M_true'][i]/(4*np.pi*500*rho_c_z))**(1/3)
        r500 /= cosmology['h']
        if configMod.profile_shape == 'BETA':
            # Build a BETA profile with BETA=2/3 (because that easy to integrate)
            # and random r500/7 < rc < r500/3
            # Note that the amplitude doesn't matter because this thing gets rescaled below
            rcFactor = 3 + 4*rng.random()
            rc = r500/rcFactor
            Micm = rc**2 * (rArr - rc*np.arctan(rArr/rc))
            # Normalize it such that Micm(r500) = mock[i,3]
            Micmr500 = np.interp(r500, rArr, Micm)
            Micm *= mock[configMod.Xray_obs][i]/Micmr500
        elif configMod.profile_shape == 'PL':
            Micm = mock[configMod.Xray_obs][i] * (rArr/r500)**(scaling['slope_MgR'] + scaling['slope_MgR_std']*rng.standard_normal())
        # Scale back to ref cosmology
        Micmref = Micm * (dAref/dA)**2.5
        # Write to catalog
        mock['%s_profile' % configMod.Xray_obs][i][0] = r_ref
        mock['%s_profile' % configMod.Xray_obs][i][1] = Micmref

    # False detections
    mock['REDSHIFT_LIMIT'] = np.zeros(nCluster)
    mock['REDSHIFT_LIMIT'][mock['REDSHIFT'] == 0.] = 1.4

    # Redshift errors
    mock['REDSHIFT_UNC'] = np.zeros(nCluster)

    # M500 estimate, neede for defining X-ray observable
    mock['M500'] = rng.lognormal(np.log(mock['M_true']/cosmology['h']), scaling['Dsz']/scaling['Bsz'])

    # Theta_core (random)
    theta_core_Mpc = rng.exponential(scale=1/3.76, size=len(mock))
    theta_core_arcmin = theta_core_Mpc*cosmology['h'] / [cosmo.dA(z, cosmology) for z in mock['REDSHIFT']] * 180/np.pi * 60
    theta_core = np.round(theta_core_arcmin*4)/4
    theta_core[theta_core > 3] = 3.
    mock['THETA_CORE'] = theta_core

    # Sample
    mock['COSMO_SAMPLE'] = np.ones(nCluster, dtype=int)

    # Save to fits
    if catalog_name is None:
        catalog_name = 'mock_%s.fits' % time.strftime("%y%m%d-%H%M%S")
    mock.write(catalog_name)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        tmp = None
    else:
        tmp = sys.argv[2]
    main(sys.argv[1], tmp)
