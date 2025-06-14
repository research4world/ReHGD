import numpy as np
from numba import njit, prange

nf4_data = np.array([-1.0,
                     -0.6961928009986877,
                     -0.5250730514526367,
                     -0.39491748809814453,
                     -0.28444138169288635,
                     -0.18477343022823334,
                     -0.09105003625154495,
                     0.0,
                     0.07958029955625534,
                     0.16093020141124725,
                     0.24611230194568634,
                     0.33791524171829224,
                     0.44070982933044434,
                     0.5626170039176941,
                     0.7229568362236023,
                     1.0], dtype=np.float32)


@njit(cache=True)
def quantize_2d_nf4(x):
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            if x[i][j] > 0.03979014977812767:
                if x[i][j] > 0.3893125355243683:
                    if x[i][j] > 0.6427869200706482:
                        if x[i][j] > 0.8614784181118011:
                            x[i][j] = 0b1111
                        else:
                            x[i][j] = 0b1110
                    else:
                        if x[i][j] > 0.5016634166240692:
                            x[i][j] = 0b1101
                        else:
                            x[i][j] = 0b1100
                else:
                    if x[i][j] > 0.2035212516784668:
                        if x[i][j] > 0.2920137718319893:
                            x[i][j] = 0b1011
                        else:
                            x[i][j] = 0b1010
                    else:
                        if x[i][j] > 0.1202552504837513:
                            x[i][j] = 0b1001
                        else:
                            x[i][j] = 0b1000
            else:
                if x[i][j] > -0.33967943489551544:
                    if x[i][j] > -0.13791173323988914:
                        if x[i][j] > -0.045525018125772476:
                            x[i][j] = 0b0111
                        else:
                            x[i][j] = 0b0110
                    else:
                        if x[i][j] > -0.23460740596055984:
                            x[i][j] = 0b0101
                        else:
                            x[i][j] = 0b0100
                else:
                    if x[i][j] > -0.6106329262256622:
                        if x[i][j] > -0.4599952697753906:
                            x[i][j] = 0b0011
                        else:
                            x[i][j] = 0b0010
                    else:
                        if x[i][j] > -0.8480964004993439:
                            x[i][j] = 0b0001
                        else:
                            x[i][j] = 0b0000

    return x


def get_scale(mat):
    m_max = np.abs(mat).max()
    scale = m_max if m_max > 0. else 1.

    return scale


def quantize(mat, scale):
    mat = mat / scale
    qmat = quantize_2d_nf4(mat)
    qmat = qmat.astype(np.uint8)

    return qmat


def dequantize(qmat, scale, dtype):
    mat = nf4_data[qmat] * scale
    mat = mat.astype(dtype)

    return mat
