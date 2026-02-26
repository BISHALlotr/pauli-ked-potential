"""
Numerical radial derivatives via a Lagrange interpolation method.

Provides:
- radial gradient
- radial divergence
- radial Laplacian
- vector derivatives: grad, lapl, grad(lapl), lapl(lapl)

Notes
-----
This module assumes `localderivs(dens, grid, nn, mm=0)` is implemented in
`pauliked.interpolator` (or similar) and returns first derivative and optionally
higher derivatives.
"""

from __future__ import annotations
import numpy as np
from . import interpolator

__all__ = [
    "RadialDivergence",
    "RadialGradient",
    "RadialLaplacian",
    "Vectorderivs",
    # snake_case aliases:
    "radial_divergence",
    "radial_gradient",
    "radial_laplacian",
    "vector_derivs",
]


def RadialDivergence(dens, grid, nn, mm=0):
    """
    Numerical radial divergence of a radial function.

    Parameters
    ----------
    dens : array_like
        Values of the radial function on the grid.
    grid : array_like
        Radial grid points.
    nn : int
        Interpolation stencil length.
    mm : int, optional
        Binomial convolution half-width (default: 0).

    Returns
    -------
    div : ndarray
        Divergence evaluated on the grid.
    """
    dens = np.asarray(dens)
    grid = np.asarray(grid)

    deriv1, _ = interpolator.localderivs(dens, grid, nn, mm)
    div = deriv1 + 2.0 * dens / grid
    return div


def RadialGradient(dens, grid, nn, mm=0):
    """
    Numerical radial gradient of a radial function.

    Returns the first derivative with respect to the radial coordinate.
    """
    dens = np.asarray(dens)
    grid = np.asarray(grid)

    deriv1, _ = interpolator.localderivs(dens, grid, nn, mm)
    return deriv1


def RadialLaplacian(dens, grid, nn, mm=0):
    """
    Numerical radial Laplacian of a radial function.

    Computed as divergence(gradient(f)).
    """
    dens = np.asarray(dens)
    grid = np.asarray(grid)

    deriv1, _ = interpolator.localderivs(dens, grid, nn, mm)
    lapl = RadialDivergence(deriv1, grid, nn, mm)
    return lapl


def Vectorderivs(dens, grid, nn, useiorder=1):
    """
    Compute radial grad, lapl, grad(lapl), and lapl(lapl) on a radial grid.

    Parameters
    ----------
    dens : array_like
        Density/function values.
    grid : array_like
        Radial grid values.
    nn : int
        Interpolation stencil length.
    useiorder : int, optional
        If 2, use a direct second-derivative route; otherwise build recursively.

    Returns
    -------
    grad, lapl, grlapl, lalapl : ndarray
        Arrays of the derivatives on the grid.
    """
    dens = np.asarray(dens)
    grid = np.asarray(grid)

    if useiorder == 2:
        deriv1, deriv2 = interpolator.localderivs(dens, grid, nn)
        deriv3, deriv4 = interpolator.localderivs(deriv2, grid, nn)
    else:
        deriv1, _ = interpolator.localderivs(dens, grid, nn)
        deriv2, _ = interpolator.localderivs(deriv1, grid, nn)
        deriv3, _ = interpolator.localderivs(deriv2, grid, nn)
        deriv4, _ = interpolator.localderivs(deriv3, grid, nn)

    grad = deriv1
    lapl = deriv2 + 2.0 * deriv1 / grid
    grlapl = deriv3 + 2.0 * deriv2 / grid - 2.0 * deriv1 / (grid**2)
    lalapl = deriv4 + 4.0 * deriv3 / grid
    return grad, lapl, grlapl, lalapl


# --- production-style snake_case aliases (do NOT break old code) ---
def radial_divergence(dens, grid, nn, mm=0):
    return RadialDivergence(dens, grid, nn, mm)


def radial_gradient(dens, grid, nn, mm=0):
    return RadialGradient(dens, grid, nn, mm)


def radial_laplacian(dens, grid, nn, mm=0):
    return RadialLaplacian(dens, grid, nn, mm)


def vector_derivs(dens, grid, nn, useiorder=1):
    return Vectorderivs(dens, grid, nn, useiorder)
