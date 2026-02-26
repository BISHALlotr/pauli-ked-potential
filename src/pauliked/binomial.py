"""Create a normalized binomial kernel and convolve a 1D grid function with it."""
import numpy


def binomial(m: int):
    """Return normalized binomial weights of order m (length m+1)."""
    b = numpy.zeros(m + 1)
    b[0] = 1.0 / (2**m)
    for i in range(m):
        b[i + 1] = b[i] * (m - i) / (i + 1)
    return b


def gridconvolve(F, B):
    """Convolve 1D array F with kernel B using zero-padding at both ends."""
    ndim = len(F)
    mdim = len(B)
    mm = mdim // 2

    Fpadded = numpy.concatenate((numpy.zeros(mm), F, numpy.zeros(mm)))
    Fconvolved = numpy.zeros(ndim)

    for i in range(ndim):
        Fconvolved[i] = numpy.add.reduce(Fpadded[i : i + mdim] * B)

    return Fconvolved


if __name__ == "__main__":
    B = binomial(6)
    print("# calculated", B)
    print("# exact", 1.0 / 2**6, 6.0 / 2**6, 15.0 / 2**6, 20.0 / 2**6, 15.0 / 2**6, 6.0 / 2**6, 1.0 / 2**6)

    x = numpy.arange(2.0, 4.0, 0.01)
    x0 = 3.0
    gamm = 0.05
    funk = (1 / numpy.pi) * 0.5 * gamm / ((x - x0) ** 2 + 0.25 * gamm**2)

    for m in (10, 20, 40, 80):
        B = binomial(m)
        convolved = gridconvolve(funk, B)
        # (printing a ton of lines is fine; keeping your original loop is OK)