"""
cmb_distance_prior_like_5d.py

Generalized (N-dimensional) version of cmb_distance_prior_like.py. Reads
a mean vector + covariance matrix from a data file (produced by
derive_cmb_distance_priors.py) rather than individual ini options, so it
scales to the 5D vector (R, l_A, ombh2, ns, 1e9*As) instead of just 3D.

R and l_A still come from pre_cmb.py's 'cmb_distance_priors' section
(the geo-split quantities). ombh2, ns, and A_s are read directly from
cosmological_parameters, same pattern as the 3D version - no split
applies to them, they are ordinary free parameters already in your
cosmology.

Ini options:
    data_file   - path to the mean+covariance file (see file format
                  written by derive_cmb_distance_priors.py)
    feedback    - optional, default False

Data file format (matches derive_cmb_distance_priors.py's output):
    # order: R l_A ombh2 ns 1e9*As
    # mean vector
    <5 numbers>
    # covariance matrix
    <5 numbers>
    <5 numbers>
    <5 numbers>
    <5 numbers>
    <5 numbers>
"""
import numpy as np
from cosmosis.datablock import names, option_section

cosmo_section = names.cosmological_parameters
cmb_priors = "cmb_distance_priors"
likes = names.likelihoods


def _load_mean_cov(path):
    with open(path) as f:
        lines = [l for l in f if not l.strip().startswith("#") and l.strip()]
    mean = np.array([float(v) for v in lines[0].split()])
    cov_rows = [np.array([float(v) for v in l.split()]) for l in lines[1:1 + len(mean)]]
    cov = np.vstack(cov_rows)
    return mean, cov


def setup(options):
    section = option_section
    data_file = options.get_string(section, "data_file")
    feedback = options.get_bool(section, "feedback", default=False)

    mean, cov = _load_mean_cov(data_file)
    inv_cov = np.linalg.inv(cov)

    if mean.shape[0] != 5:
        raise ValueError(
            f"Expected a 5D data vector (R, l_A, ombh2, ns, 1e9*As), "
            f"got {mean.shape[0]} entries in {data_file}"
        )

    return mean, inv_cov, feedback


def execute(block, config):
    mean, inv_cov, feedback = config

    R = block[cmb_priors, "R"]
    l_A = block[cmb_priors, "l_A"]
    ombh2 = block[cosmo_section, "ombh2"]
    ns = block[cosmo_section, "n_s"]
    # Check your consistency/values setup: this may be stored as "A_s"
    # directly, or as "log1e10As" if that parameterization is used -
    # adjust the key/conversion below to match.
    As = block[cosmo_section, "A_s"]

    x = np.array([R, l_A, ombh2, ns, 1.0e9 * As])
    delta = x - mean

    chi2 = float(delta @ inv_cov @ delta)
    like = -0.5 * chi2

    block[likes, "cmb_distance_prior_like"] = like

    if feedback:
        print("R predicted       = ", R)
        print("l_A predicted     = ", l_A)
        print("ombh2 predicted   = ", ombh2)
        print("ns predicted      = ", ns)
        print("1e9*As predicted  = ", 1.0e9 * As)
        print("chi2              = ", chi2)

    return 0


def cleanup(config):
    pass
