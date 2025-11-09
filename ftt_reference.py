#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reference.py — Fractal–Topological Transform (FTT) — Reference Implementation

Author: Mohamed Orhan Zeinel (Independent Researcher)
Contact: mohamedorhanzeinel@gmail.com
Project: Fractal–Topological Transform (FTT)
Repo: https://github.com/mohamedorhan/Fractal-Topological-Transform-FTT-

License: MIT
---------------------------------------------------------------------------
This file provides a clean, single-file, CPU-only reference for FTT:
  • Deterministic fractal/bit-reversal ordering (π) for hierarchical pairing
  • Orthonormal 2-tap lifting (Haar-style) with in-place even/odd processing
  • Forward / Inverse transforms with exact round-trip (within fp tolerance)
  • Energy preservation check and reconstruction tests
  • FTT-domain blockwise product as an approximate fast convolution
  • Synthetic benchmarks (1D) with robust statistics
  • Reproducibility (fixed seeds), CLI, and clear type hints

Notes
-----
1) This is a *reference* implementation focused on correctness, clarity,
   and scientific reproducibility. It is intentionally dependency-light:
   only NumPy is required. (SciPy/NetworkX/PyTorch integrations can live
   in the repository as optional modules.)
2) The transform here uses a rigorous orthonormal, 2-tap lifting step
   equivalent to the Haar transform per level after an ordering π. That
   makes proofs and energy checks simple while keeping the structure
   consistent with the FTT paper (fractally ordered lifting, in place).
3) The “FTT-domain product” included here demonstrates the generalized
   convolution idea (blockwise multiplication aligned with scales). For
   production research, you will specialize the block structure and
   filters to the task/domain (images, graphs, CNN layers, etc.).

Usage
-----
$ python reference.py demo          # small demo + checks
$ python reference.py bench         # 1D synthetic timing across N
$ python reference.py conv_demo     # shows FTT-domain product vs. FFT conv
$ python reference.py validate      # strict numerical validations

All updates, extended experiments, and GPU/graph variants:
→ https://github.com/mohamedorhan/Fractal-Topological-Transform-FTT-
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 1337) -> None:
    """Set global RNG seed for reproducibility."""
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Utilities: powers of two, bit-reversal permutation (a classic fractal order)
# ---------------------------------------------------------------------------

def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def next_pow2(n: int) -> int:
    """Return the next power of two >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def bit_reverse(i: int, bits: int) -> int:
    """Reverse the lowest `bits` bits of integer i."""
    r = 0
    for _ in range(bits):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


def bit_reversal_permutation(N: int) -> np.ndarray:
    """
    Classic bit-reversal order used in FFT butterflies. For power-of-two N.
    If N is not a power-of-two, we return identity (caller can pad).
    """
    if not is_pow2(N):
        return np.arange(N, dtype=np.int64)
    bits = (N - 1).bit_length()
    return np.fromiter((bit_reverse(i, bits) for i in range(N)), dtype=np.int64, count=N)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FTTConfig:
    """
    Configuration for the reference FTT.
    """
    levels: Optional[int] = None         # if None → auto = floor(log2(N)) - 0
    use_bit_reversal: bool = True        # True: π = bit-reversal; False: π = identity
    orthonormal_scaling: bool = True     # ensure energy preservation per level
    safety_checks: bool = True           # run extra assertions (slower)


# ---------------------------------------------------------------------------
# Core FTT class (1D reference)
# ---------------------------------------------------------------------------

class FTT:
    """
    Reference Fractal–Topological Transform (1D, CPU, NumPy).

    We apply a permutation π (default: bit-reversal) once, then perform J levels
    of 2-tap orthonormal lifting (Haar-style) in place:

        s = (even + odd)/sqrt(2)
        d = (odd - even)/sqrt(2)

    We store (d_J, ..., d_1, s_0) when flattened; the structured API returns a
    list [d_1, d_2, ..., d_J] (fine→coarse) and s_0, matching the paper’s
    notation.

    Perfect reconstruction holds up to floating-point tolerance.
    """

    def __init__(self, N: int, config: Optional[FTTConfig] = None) -> None:
        if config is None:
            config = FTTConfig()
        object.__setattr__(self, "config", config)

        self.N_original = int(N)
        self.N = self.N_original
        self.padded = False

        if not is_pow2(self.N_original):
            # Pad to next power-of-two for clean dyadic multiresolution
            self.N = next_pow2(self.N_original)
            self.padded = True

        # Determine levels:
        auto_levels = int(math.log2(self.N))
        if config.levels is None:
            self.J = auto_levels
        else:
            self.J = int(config.levels)
            if self.J > auto_levels:
                raise ValueError(
                    f"levels={self.J} exceeds allowed max {auto_levels} for N={self.N}"
                )

        # Build permutation π and its inverse:
        if config.use_bit_reversal and is_pow2(self.N):
            self.pi = bit_reversal_permutation(self.N)
        else:
            self.pi = np.arange(self.N, dtype=np.int64)
        self.pi_inv = np.argsort(self.pi)

        # Internal check:
        if config.safety_checks:
            assert self.pi.shape == (self.N,)
            assert np.array_equal(np.sort(self.pi), np.arange(self.N))

    # --------------------------- forward / inverse ---------------------------

    def forward(self, x: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Forward FTT.

        Parameters
        ----------
        x : (N_original,) array_like
            Input real signal. If not power-of-two, we zero-pad to N.

        Returns
        -------
        d_list : list of arrays
            Detail coefficients [d_1, d_2, ..., d_J] (fine → coarse).
        s0 : array
            Coarsest residual (approximation) at level 0.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("x must be 1D")

        if x.shape[0] != self.N:
            # pad if needed
            x_pad = np.zeros(self.N, dtype=float)
            x_pad[: min(self.N_original, x.shape[0])] = x[: min(self.N_original, self.N)]
            x = x_pad

        # Permute
        x = x[self.pi]

        # Work buffer
        buf = x.copy()
        d_list: List[np.ndarray] = []

        # Level loop (fine→coarse)
        size = self.N
        for j in range(self.J, 0, -1):
            half = size // 2
            even = buf[:half]
            odd = buf[half:size]

            if self.config.orthonormal_scaling:
                # Orthonormal 2-tap Haar lifting per pair
                s = (even + odd) * (1.0 / math.sqrt(2.0))
                d = (odd - even) * (1.0 / math.sqrt(2.0))
            else:
                # Biorthogonal (not normalized)
                d = odd - even
                s = even + 0.5 * d

            d_list.append(d.copy())
            # Coarsen: next level operates on s
            buf = s
            size = half

        s0 = buf.copy()
        d_list.reverse()  # [d_1, ..., d_J]

        if self.config.safety_checks:
            # Sanity: total number of coefficients equals N
            total = s0.size + sum(d.size for d in d_list)
            assert total == self.N, "Coefficient count mismatch"
        return d_list, s0

    def inverse(self, d_list: Sequence[np.ndarray], s0: np.ndarray) -> np.ndarray:
        """
        Inverse FTT.

        Parameters
        ----------
        d_list : list [d_1, ..., d_J]
        s0     : coarsest residual

        Returns
        -------
        x_rec : reconstructed signal of length N (padded if N!=N_original)
        """
        if len(d_list) != self.J:
            raise ValueError(f"Expected {self.J} detail levels, got {len(d_list)}")

        x = s0.copy()
        # Go coarse→fine
        for j in range(1, self.J + 1):
            d = d_list[j - 1]
            half = x.size

            if self.config.orthonormal_scaling:
                # Inverse of orthonormal 2-tap
                even = (x - d) * (1.0 / math.sqrt(2.0))
                odd = (x + d) * (1.0 / math.sqrt(2.0))
            else:
                even = x - 0.5 * d
                odd = d + even

            x = np.concatenate([even, odd], axis=0)

        # Undo permutation
        x = x[self.pi_inv]

        # Truncate padding if any
        if self.padded:
            x = x[: self.N_original].copy()
        return x

    # --------------------------- convenience API ----------------------------

    def transform_flat(self, x: np.ndarray) -> np.ndarray:
        """
        Flattened coefficient vector: [d_J ... d_1 | s0] in one array.
        Useful for debugging and storing.
        """
        d_list, s0 = self.forward(x)
        # Concatenate as [d_J...d_1, s0] to keep coarse summary at the end.
        d_rev = d_list[::-1]
        return np.concatenate([*d_rev, s0], axis=0)

    def inverse_from_flat(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Reconstruct from flattened vector produced by `transform_flat`.
        """
        # sizes: N/2 + N/4 + ... + 1 = N - 1, plus s0 (1) = N total
        if coeffs.size != self.N:
            raise ValueError("coeff length must equal padded N")
        sizes = [self.N // (2 ** i) // 2 for i in range(self.J)]  # [N/2, N/4, ...]
        idx = 0
        d_rev = []
        for s in sizes:
            d_rev.append(coeffs[idx: idx + s])
            idx += s
        s0 = coeffs[idx:]
        d_list = d_rev[::-1]
        return self.inverse(d_list, s0)

    # --------------------------- energy & checks -----------------------------

    @staticmethod
    def energy(x: np.ndarray) -> float:
        """L2 energy."""
        x = np.asarray(x, dtype=float)
        return float(np.dot(x, x))

    def check_energy_preservation(self, x: np.ndarray, tol: float = 1e-8) -> float:
        """
        Compute ||x||^2 − ||coeffs||^2 (should be ~0 for orthonormal scaling).
        Returns the absolute difference.
        """
        d_list, s0 = self.forward(x)
        coeff_energy = float(np.dot(s0, s0)) + sum(float(np.dot(d, d)) for d in d_list)
        diff = abs(self.energy(x if not self.padded else np.pad(
            x, (0, self.N - self.N_original))) - coeff_energy)
        if self.config.safety_checks and diff > max(tol, 1e-12):
            # Not raising; return diff so caller can log/report
            pass
        return diff

    def roundtrip_error(self, x: np.ndarray) -> float:
        """||x - inverse(forward(x))||_2."""
        d_list, s0 = self.forward(x)
        x_rec = self.inverse(d_list, s0)
        err = float(np.linalg.norm(x[: self.N_original] - x_rec))
        return err

    # --------------------------- FTT "convolution" ---------------------------

    def product_ftt_domain(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Demonstration of generalized convolution via blockwise FTT-domain product.

        This computes:
            X = FTT(x), H = FTT(h)
            Z blocks = elementwise products per level (and coarse)
            z = FTT^{-1}(Z)

        This is *not* classical circular convolution; it is the FTT-space
        multiplication sketched in the paper. Task-specific block designs can
        tighten approximation bounds. Here we use strict 1-1 alignment.
        """
        dx, s0x = self.forward(x)
        dh, s0h = self.forward(h)

        # level-wise elementwise product
        dz = [a * b for a, b in zip(dx, dh)]
        s0z = s0x * s0h

        z = self.inverse(dz, s0z)
        return z


# ---------------------------------------------------------------------------
# Synthetic signals and kernels (for demos/benchmarks)
# ---------------------------------------------------------------------------

def gaussian_kernel_1d(N: int, sigma: float) -> np.ndarray:
    """
    Centered 1D Gaussian (not normalized as probability density;
    we normalize to unit L1 for convolution-like behavior).
    """
    x = np.arange(N)
    mu = (N - 1) / 2.0
    g = np.exp(-0.5 * ((x - mu) / float(sigma)) ** 2)
    g /= max(np.sum(g), 1e-12)
    return g


def piecewise_smooth_signal(N: int, jumps: int = 4, noise: float = 0.05) -> np.ndarray:
    """
    Piecewise-smooth synthetic signal with a few random jumps.
    """
    set_seed(1234)
    x = np.zeros(N, dtype=float)
    segs = np.linspace(0, N, jumps + 2, dtype=int)
    for s0, s1 in zip(segs[:-1], segs[1:]):
        slope = np.random.uniform(-1.0, 1.0)
        base = np.random.uniform(-1.0, 1.0)
        t = np.arange(s0, s1)
        x[s0:s1] = base + slope * (t - s0) / max(s1 - s0, 1)
    if noise > 0:
        x += noise * np.random.randn(N)
    return x


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def benchmark_ftt_times(N_list: Iterable[int], trials: int = 10) -> List[Tuple[int, float, float]]:
    """
    Benchmark forward+inverse timing (median, MAD) for each N in N_list.
    Returns list of tuples: (N, median_ms, mad_ms).
    """
    results = []
    set_seed(2025)
    for N in N_list:
        ftt = FTT(N)
        times_ms: List[float] = []
        for _ in range(trials):
            x = piecewise_smooth_signal(N, jumps=6, noise=0.02)
            t0 = time.perf_counter()
            d, s0 = ftt.forward(x)
            _ = ftt.inverse(d, s0)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1e3)
        median = statistics.median(times_ms)
        mad = statistics.median([abs(t - median) for t in times_ms])
        results.append((N, median, mad))
    return results


# ---------------------------------------------------------------------------
# Demos & Validation
# ---------------------------------------------------------------------------

def demo() -> None:
    """Small end-to-end demonstration with checks."""
    N = 4096
    print(f"[demo] N={N}")
    ftt = FTT(N)
    x = piecewise_smooth_signal(N, jumps=5, noise=0.03)
    diff = ftt.check_energy_preservation(x)
    err = ftt.roundtrip_error(x)
    print(f"  energy diff: {diff:.3e}")
    print(f"  round-trip error: {err:.3e}")

    # Show first few coefficients sizes
    d_list, s0 = ftt.forward(x)
    sizes = [d.size for d in d_list] + [s0.size]
    print("  coeff sizes (d1..dJ, s0):", sizes)


def demo_conv() -> None:
    """Demonstrate FTT-domain product vs. FFT circular convolution."""
    N = 8192
    x = piecewise_smooth_signal(N, jumps=6, noise=0.02)
    h = gaussian_kernel_1d(N, sigma=32)

    ftt = FTT(N)
    z_ftt = ftt.product_ftt_domain(x, h)

    # FFT circular convolution (for reference)
    X = np.fft.rfft(x)
    H = np.fft.rfft(h)
    z_fft = np.fft.irfft(X * H, n=N)

    # Compare shapes (they are different operations; we just report norms)
    print(f"[conv_demo] N={N}")
    print(f"  ||z_ftt||2 = {np.linalg.norm(z_ftt):.6f}")
    print(f"  ||z_fft||2 = {np.linalg.norm(z_fft):.6f}")
    print(f"  ||z_ftt - z_fft||2 = {np.linalg.norm(z_ftt - z_fft):.6f}")


def validate(strict: bool = True) -> None:
    """
    Run numerical validations: energy preservation and roundtrip error across sizes.
    """
    Ns = [1024, 2048, 4096, 8192]
    tol_energy = 1e-8
    tol_roundtrip = 1e-9
    ok = True

    for N in Ns:
        ftt = FTT(N)
        x = piecewise_smooth_signal(N, jumps=5)
        diff = ftt.check_energy_preservation(x, tol=tol_energy)
        err = ftt.roundtrip_error(x)
        print(f"[validate] N={N}  energy diff={diff:.3e}  round-trip={err:.3e}")
        if strict and (diff > tol_energy * 10 or err > tol_roundtrip * 10):
            ok = False

    if strict and not ok:
        raise AssertionError("Validation failed; see printed diffs above.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fractal–Topological Transform (FTT) — Reference Implementation"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="Small demo with energy/roundtrip checks.")
    sub.add_parser("conv_demo", help="Demonstrate FTT-domain product vs. FFT.")
    p_bench = sub.add_parser("bench", help="Benchmark forward+inverse over N.")
    p_bench.add_argument(
        "--sizes",
        type=str,
        default="4096,16384,65536",
        help="Comma-separated sizes to test (power-of-two), e.g. 4096,16384,65536",
    )
    p_bench.add_argument("--trials", type=int, default=10, help="Trials per size.")
    p_bench.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional path to save CSV results (N,median_ms,MAD_ms).",
    )

    p_val = sub.add_parser("validate", help="Run numerical validations.")
    p_val.add_argument("--loose", action="store_true", help="Non-strict thresholds.")

    args = parser.parse_args()

    if args.cmd == "demo":
        demo()
    elif args.cmd == "conv_demo":
        demo_conv()
    elif args.cmd == "bench":
        sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
        results = benchmark_ftt_times(sizes, trials=args.trials)
        print("N, median_ms, MAD_ms")
        for N, med, mad in results:
            print(f"{N}, {med:.3f}, {mad:.3f}")
        if args.csv:
            os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
            with open(args.csv, "w", encoding="utf-8") as f:
                f.write("N,median_ms,MAD_ms\n")
                for N, med, mad in results:
                    f.write(f"{N},{med:.6f},{mad:.6f}\n")
            print(f"[bench] saved: {args.csv}")
    elif args.cmd == "validate":
        validate(strict=not args.loose)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
