"""
von Weizsäcker-like (vW) enhancement factor piece used in deorbitalization contexts.

Reference:
Daniel Mejia-Rodriguez, S. B. Trickey, Phys. Rev. A 96, 052512 (2017), Eq. (27)
"""

from __future__ import annotations
import numpy as np

__all__ = ["FvW"]


def FvW(s_sqr, q):
    """
    Parameters
    ----------
    s_sqr : array_like
        Reduced gradient squared.
    q : array_like
        Reduced Laplacian (unused here; included for unified functional API).

    Returns
    -------
    f, dfds2, dfdq : ndarray
    """
    s_sqr = np.asarray(s_sqr)
    one = np.ones_like(s_sqr, dtype=float)

    f = (5.0 / 3.0) * s_sqr
    dfds2 = (5.0 / 3.0) * one
    dfdq = np.zeros_like(one)

    return f, dfds2, dfdq