import numpy
from . import fundpoly
from . import binomial


def gridinterval(index, ir, nn):
    """
    Calculate interval limits about a given point.

    Inputs:
      index: length of array over which ir ranges
      ir: integer index of current point
      nn: half-width of interval about point

    Returns:
      (lower_offset, upper_offset) relative to ir
    """
    return max(0, ir - nn) - ir, min(index - 1, ir + nn) - ir


def lightweights(s, rmesh, iorder=1):
    """
    Fast weight construction for local interpolation derivatives.

    Returns:
      H: weights for y'(x)  so y'(x_s) ~ sum_i H[i] * y_i
      J: weights for y''(x) so y''(x_s) ~ sum_i J[i] * y_i  (zeros if iorder==1)
    """
    Nbatch = len(rmesh)

    zero = 0.0
    one = 1.0
    two = 2.0

    pisubi = numpy.zeros((Nbatch))  # subpolynomials used for weights
    Gp = numpy.zeros((Nbatch))      # derivative coeffs (w_i')
    Gpp = numpy.zeros((Nbatch))     # second-derivative coeffs (w_i'')

    H = numpy.zeros((Nbatch))       # weights for y'(x)
    J = numpy.zeros((Nbatch))       # weights for y''(x)

    indices = numpy.arange(Nbatch)

    # Denominator polynomials independent of s
    for i in range(Nbatch):
        pii = i - indices
        pii[i] = 1
        pisubi[i] = numpy.multiply.reduce(pii)

    pisubs = pisubi[s]

    # Oddball factor with diagonal form
    pis = s - indices
    pis[s] = one
    ipis = one / pis
    ipis[s] = zero

    sumthing = numpy.add.reduce(ipis)

    # First-derivative weight factors
    Gp = (pisubs / pisubi) * ipis
    Gp[s] = sumthing

    # x'(s)
    xp = numpy.add.reduce(rmesh * Gp)

    # Chain rule for y'(x)
    H = Gp / xp

    # Second derivative if requested
    if iorder == 2:
        sumsqthing = numpy.add.reduce(ipis * ipis)

        Gpp = two * Gp * (sumthing - one * ipis)
        Gpp[s] = sumthing * sumthing - sumsqthing

        # x''(s)
        xpp = numpy.add.reduce(rmesh * Gpp)

        # Chain rule for y''(x)
        J = (Gpp - H * xpp) / (xp**2)

    return H, J


def weights(s, rmesh):
    """
    Slower (reference) weight computation using fundpoly.poly().
    Returns H, J weights for first and second derivatives.
    """
    N = len(rmesh)

    Gp = numpy.zeros((N))
    Gpp = numpy.zeros((N))

    H = numpy.zeros((N))
    J = numpy.zeros((N))

    for i in range(N):
        k = fundpoly.poly(N, [i], i)

        if i == s:
            for j in range(N):
                if j != i:
                    Gp[i] = Gp[i] + 1.0 / (s - j)
                    for l in range(N):
                        if (l != i) and (l > j):
                            Gpp[i] = Gpp[i] + 2.0 * fundpoly.poly(N, [i, j, l], s) / k
        else:
            Gp[i] = fundpoly.poly(N, [i, s], s) / k
            for j in range(N):
                if (j != s) and (j != i):
                    Gpp[i] = Gpp[i] + 2.0 * fundpoly.poly(N, [i, s, j], s) / k

    xp = 0.0
    xpp = 0.0
    for j in range(N):
        xp = xp + rmesh[j] * Gp[j]
        xpp = xpp + rmesh[j] * Gpp[j]

    H = Gp / xp
    J = (Gpp - H * xpp) / (xp**2)

    return H, J


def localderivs(dens, rmesh, nn, mmconvolve=0, iorder=1):
    """
    Compute local first and (optionally) second derivatives of dens along rmesh.

    WARNING: This is NOT the spherical/radial gradient or Laplacian operator.
             It's d/dx and d^2/dx^2 along the mesh coordinate.

    Inputs:
      dens: 1d array values
      rmesh: 1d array mesh positions (same length as dens)
      nn: locality parameter (stencil half-width)
      mmconvolve: if >0, smooth grad/lapl with binomial convolution after compute
      iorder: 1 or 2 (compute first only or first+second)

    Returns:
      grad, lapl
    """
    if rmesh.size != dens.size:
        raise ValueError("rmesh and dens must have the same size")

    if iorder not in (1, 2):
        raise ValueError("iorder must be 1 or 2")

    N = dens.size
    grad = numpy.zeros((N))
    lapl = numpy.zeros((N))

    for x in range(N):
        in1, in2 = gridinterval(N, x, nn)
        nbatch = dens[x + in1 : x + in2 + 1]
        rbatch = rmesh[x + in1 : x + in2 + 1]

        H, J = lightweights(-in1, rbatch, iorder=iorder)

        grad[x] = numpy.add.reduce(H * nbatch)
        lapl[x] = numpy.add.reduce(J * nbatch)

    # Optional smoothing of derivative arrays
    if mmconvolve > 0:
        B = binomial.binomial(2 * mmconvolve)
        grad = binomial.gridconvolve(grad, B)
        lapl = binomial.gridconvolve(lapl, B)

    return grad, lapl