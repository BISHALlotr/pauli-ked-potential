"""
Dimensionless reduced variables used in KED/meta-GGA style functionals.

s^2 = |∇n|^2 / (4 k_F^2 n^2)
q   = (∇^2 n / n^(5/3)) / (4 (3π^2)^(2/3))
"""

from __future__ import annotations
import numpy as np

__all__ = ["get_s_sqr", "get_q"]


def get_s_sqr(n, dn_by_dr):
    """
    Parameters
    ----------
    n : array_like
        Electron density.
    dn_by_dr : array_like
        Radial derivative of density.

    Returns
    -------
    s_sqr : ndarray
    s_sqr_n : ndarray
        ∂(s^2)/∂n
    s_sqr_deln : ndarray
        ∂(s^2)/∂(|∇n|)  (here |∇n| is represented by dn_by_dr in radial case)
    """
    n = np.asarray(n)
    dn_by_dr = np.asarray(dn_by_dr)

    # Avoid divide-by-zero blowups while keeping vectorization
    eps = np.finfo(float).tiny
    n_safe = np.maximum(n, eps)
    dn_safe = np.where(dn_by_dr == 0.0, eps, dn_by_dr)

    kf = (3.0 * np.pi**2 * n_safe) ** (1.0 / 3.0)
    del_n_atom = (dn_by_dr**2) / (n_safe**2)

    s_sqr = del_n_atom / (4.0 * kf**2)

    # Your original formulas preserved:
    s_sqr_n = (-8.0 / 3.0) * s_sqr / n_safe
    s_sqr_deln = 2.0 * s_sqr / dn_safe

    return s_sqr, s_sqr_n, s_sqr_deln


def get_q(n, lapn):
    """
    Parameters
    ----------
    n : array_like
        Electron density.
    lapn : array_like
        Radial Laplacian of density.

    Returns
    -------
    q : ndarray
    q_n : ndarray
        ∂q/∂n
    q_lapn : ndarray
        ∂q/∂(∇^2 n)
    """
    n = np.asarray(n)
    lapn = np.asarray(lapn)

    eps = np.finfo(float).tiny
    n_safe = np.maximum(n, eps)
    lap_safe = np.where(lapn == 0.0, eps, lapn)

    C_factor = 4.0 * (3.0 * np.pi**2) ** (2.0 / 3.0)
    fifththird = 5.0 / 3.0

    lapn_n = lapn / (n_safe ** fifththird)
    q = lapn_n / C_factor

    q_n = (-5.0 / 3.0) * q / n_safe
    q_lapn = q / lap_safe

    return q, q_n, q_lapn