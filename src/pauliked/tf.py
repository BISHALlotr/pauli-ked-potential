"""
Thomas-Fermi kinetic energy density (uniform electron gas reference).
"""

from __future__ import annotations
import numpy as np

__all__ = ["get_tautf", "Ft"]


def get_tautf(n):
    """
    Parameters
    ----------
    n : array_like
        Electron density.

    Returns
    -------
    tautf : ndarray
        Thomas-Fermi KE density.
    d_tautf_dn : ndarray
        d(tautf)/dn
    """
    n = np.asarray(n)
    kf = (3.0 * np.pi**2 * n) ** (1.0 / 3.0)
    tautf = (3.0 / 10.0) * kf**2 * n
    d_tautf_dn = 0.5 * kf**2
    return tautf, d_tautf_dn


def Ft(s_sqr, q):
    """
    Thomas-Fermi enhancement factor: f = 1 (constant).
    Included for API consistency with other functionals: returns (f, df/ds2, df/dq).
    """
    s_sqr = np.asarray(s_sqr)
    one = np.ones_like(s_sqr, dtype=float)
    f = one
    dfds2 = np.zeros_like(one)
    dfdq = np.zeros_like(one)
    return f, dfds2, dfdq