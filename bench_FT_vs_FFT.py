# bench_FT_vs_FFT.py
import numpy as np
import time
from statistics import median
from ft import FT

N      = 100_000   # جرّب لاحقاً 262_144 و 1_048_576
ITERS  = 10        # عدد مرات القياس لالتقاط وسيط مستقر

def bench_fft(x: np.ndarray, iters: int = ITERS) -> float:
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = np.fft.fft(x)        # baseline (real-to-complex)
        times.append(time.perf_counter() - t0)
    return median(times)

def bench_ft(x: np.ndarray, levels: int = 6, iters: int = ITERS) -> float:
    ft = FT(levels=levels)
    # warm-up (خصوصاً لأول تشغيل ليتم تجميع JIT)
    _ = ft.transform(x)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = ft.transform(x)
        times.append(time.perf_counter() - t0)
    return median(times)

if __name__ == "__main__":
    print("\n=== Benchmark: FT vs FFT ===")
    # استخدم float32 للعدالة والأداء
    x = np.random.randn(N).astype(np.float32)

    t_fft = bench_fft(x, ITERS)
    t_ft  = bench_ft(x, levels=6, iters=ITERS)

    print(f"Array length: {N}")
    print(f"FFT  median over {ITERS} runs: {t_fft*1000:.3f} ms")
    print(f"FT   median over {ITERS} runs: {t_ft*1000:.3f} ms")
    print("\nDone.\n")
