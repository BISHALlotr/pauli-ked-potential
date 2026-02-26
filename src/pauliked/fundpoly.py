"""Fundamental pi-polynomial and derivatives for Lagrange interpolation"""

def poly(N, omit, s):
    """
    Returns the fundamental polynomial at a point.

    Inputs:
      N: total number of indices
      omit: list of indices to omit (subscripts i, j of pi_ij)
      s: point at which poly is evaluated

    Output:
      value: value of polynomial at s
    """
    if len(omit) >= N:
        return 0.0

    value = 1.0

    # work on a copy so we don't mutate caller's list
    omit = list(omit)

    # remove duplicates (keep first occurrence)
    for i in omit[:]:
        while omit.count(i) > 1:
            omit.remove(i)

    indices = list(range(N))  # must be a list because we call remove()

    for i in omit:
        if i in indices:       # avoid ValueError if omit has out-of-range values
            indices.remove(i)

    for j in indices:
        value *= (s - j) * 1.0

    return value