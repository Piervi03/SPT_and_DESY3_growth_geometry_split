import numpy as np

THRESHOLD = 1e-8

def set_covmats(todo, scaling, covmat):
    """Populate `covmat` dict with all possible covariance matrices between all
    observables we're currently analyzing. The scatter in velocity dispersions
    depends on cluster properties and therefore cannot be pre-computed.
    Return: (bool) whether or not all covariance matrices can be inverted (by
    checking whether all determinants are >= THRESHOLD)
    """

    ##### one follow-up observable
    if todo['Yx'] or todo['Mgas']:
        cov = [[scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
        [scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        covmat['Yx'] = cov
        covmat['Mgas'] = cov

    if todo['veldisp']:
        covmat['disp'] = None

    if todo['richness']:
        cov = [[scaling['Drichness']**2, scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness']],
            [scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        covmat['richness'] = cov

    if todo['WL']:
        cov = [[scaling['DWL_Megacam']**2, scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam']],
            [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        covmat['WLMegacam'] = cov
        cov = [[scaling['DWL_DES']**2, scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES']],
            [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        covmat['WLDES'] = cov

    ##### two follow-up observables
    if todo['Yx'] and todo['veldisp']:
        covmat['Yxdisp'] = None

    if (todo['Yx'] or todo['Mgas']) and todo['WL']:
        # Megacam
        cov = [[scaling['DWL_Megacam']**2, scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam']],
            [scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
            [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam'], scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        covmat['XrayMegacam'] = cov
        # DES
        cov = [[scaling['DWL_DES']**2, scaling['rhoWLX']*scaling['DWL_DES']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES']],
            [scaling['rhoWLX']*scaling['DWL_DES']*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
            [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES'], scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        covmat['XrayDES'] = cov

    return True


#
# def isfine(todo, scaling):
#     """Return bool whether or not all covariance matrices are invertible. We set
#     the threshold at 1e-8 because it works."""
#     ##### one follow-up observable
#     if todo['Yx'] or todo['Mgas']:
#         cov = [[scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
#             [scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
#         if np.linalg.det(cov) < 1e-8:
#             return False
#
#     if todo['veldisp']:
#         Dsigma = scaling['Ddisp0'] + scaling['DdispN']/9.
#         cov = [[Dsigma**2, scaling['rhoSZdisp']*scaling['Dsz']*Dsigma],
#             [scaling['rhoSZdisp']*scaling['Dsz']*Dsigma, scaling['Dsz']**2]]
#         if np.linalg.det(cov) < 1e-8:
#             return False
#         Dsigma = scaling['Ddisp0'] + scaling['DdispN']/89.
#         cov = [[Dsigma**2, scaling['rhoSZdisp']*scaling['Dsz']*Dsigma],
#             [scaling['rhoSZdisp']*scaling['Dsz']*Dsigma, scaling['Dsz']**2]]
#         if np.linalg.det(cov) < 1e-8:
#             return False
#
#     if todo['richness']:
#         cov = [[scaling['Dlambda']**2, scaling['rhoSZlambda']*scaling['Dsz']*scaling['Dlambda']],
#             [scaling['rhoSZlambda']*scaling['Dsz']*scaling['Dlambda'], scaling['Dsz']**2]]
#         if np.linalg.det(cov) < 1e-8:
#             return False
#
#     if todo['WL']:
#         cov = [[scaling['DWL_Megacam']**2, scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam']],
#             [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam'], scaling['Dsz']**2]]
#         if np.linalg.det(cov) < 1e-8:
#             return False
#         cov = [[scaling['DWL_DES']**2, scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES']],
#             [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES'], scaling['Dsz']**2]]
#         if np.linalg.det(cov) < 1e-8:
#             return False
#
#     ##### two follow-up observables
#     if todo['Yx'] or todo['veldisp']:
#         Dsigma = scaling['Ddisp0'] + scaling['DdispN']/9.
#         cov = [[Dsigma**2, scaling['rhoXdisp']*Dsigma*scaling['Dx'], scaling['rhoSZdisp']*scaling['Dsz']*Dsigma],
#             [scaling['rhoXdisp']*Dsigma*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
#             [scaling['rhoSZdisp']*scaling['Dsz']*Dsigma, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
#         if np.linalg.det(cov) < 1e-8:
#             return False
#         Dsigma = scaling['Ddisp0'] + scaling['DdispN']/89.
#         cov = [[Dsigma**2, scaling['rhoXdisp']*Dsigma*scaling['Dx'], scaling['rhoSZdisp']*scaling['Dsz']*Dsigma],
#                 [scaling['rhoXdisp']*Dsigma*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
#                 [scaling['rhoSZdisp']*scaling['Dsz']*Dsigma, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
#         if np.linalg.det(cov) < 1e-8:
#             return False
#
#     if (todo['Yx'] or todo['Mgas']) and todo['WL']:
#         # Megacam
#         cov = [[scaling['DWL_Megacam']**2, scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam']],
#             [scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
#             [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam'], scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
#         if np.linalg.det(cov) < 1e-8:
#             return False
#         # DES
#         cov = [[scaling['DWL_DES']**2, scaling['rhoWLX']*scaling['DWL_DES']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES']],
#             [scaling['rhoWLX']*scaling['DWL_DES']*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
#             [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES'], scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
#         if np.linalg.det(cov) < 1e-8:
#             return False
#
#     return True
