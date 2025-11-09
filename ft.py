# ft.py
import numpy as np

try:
    import numba as nb
    _HAS_NUMBA = True
except:
    _HAS_NUMBA = False

if _HAS_NUMBA:
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _nexa_kernel_nb(y, levels):
        N = y.shape[0]
        for lvl in range(levels):
            step = 1 << lvl             # 2**lvl
            block = step << 1           # 2*step
            limit = N - block
            # parallel on elements
            for i in nb.prange(0, limit+1):
                # compute index in block coordinates
                r = i % block
                if r < step:
                    a = y[i]
                    b = y[i + step]
                    s = 0.5*(a+b)
                    y[i]       = s
                    y[i+step]  = s

else:
    def _nexa_kernel_nb(y, levels):
        # fallback python
        N = y.shape[0]
        for lvl in range(levels):
            step = 1 << lvl
            block = step << 1
            limit = N - block
            for i in range(0, limit+1):
                r = i % block
                if r < step:
                    a = y[i]
                    b = y[i + step]
                    s = 0.5*(a+b)
                    y[i]       = s
                    y[i+step]  = s


class FT:
    def __init__(self, levels=6):
        self.levels = levels

    def transform(self, x):
        y = np.ascontiguousarray(x, dtype=np.float32)
        _nexa_kernel_nb(y, self.levels)
        return y
