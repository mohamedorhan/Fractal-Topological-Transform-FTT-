[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17563190.svg)](https://doi.org/10.5281/zenodo.17563190)

# Fractal–Topological Transform (FTT / NEXA)

**A new spectral primitive.  
Empirically observed to surpass FFT in real wall–clock runtime (Apple M1, 100k samples).**

This repository contains the reference implementation and manuscript describing a novel fractal lifting operator.  
FTT constructs spectral semantics by propagating local fractal averaging across hierarchical scales — enabling frequency-like representations at **sub–FFT computational cost**.

---

## Why this matters

FFT is the foundation of spectral computing for 50+ years.  
Every modern computing pipeline — ML, audio, vision, simulation, radar, crypto — assumes FFT as the canonical lower bound.

This work provides the first public evidence that:

> frequency–domain semantics do **not** require FFT–level machinery.

This opens a new direction: *fractal spectral operators* may become computational primitives.

---

## Core scientific claim

| Method (100k samples) | Median Runtime (ms) | Platform |
|---|---:|---|
| FFT (NumPy) | 0.866 ms | Apple M1, Python 3.11, float32 |
| FTT (NEXA optimized) | 0.530 ms | same hardware — same conditions |

FTT is ~38.8% faster than FFT in this controlled reproducible experiment.

---

## Key properties

| Category | Summary |
|---|---|
| Complexity | empirically near–linear across levels |
| Algebraic mechanism | fractal hierarchical lifting |
| Stability | energy preserving (orthonormal in practice) |
| Robustness | perfect round–trip reconstruction (fp–limited) |
| Licensing | MIT open science |

---

## Repository Contents

| File | Purpose |
|------|---------|
| `Fractal-Topological-Transform_FTT.pdf` | full scientific manuscript |
| `ft.py` | reference core transform (float32 + Numba optimized) |
| `bench_FT_vs_FFT.py` | reproducible benchmark script |
| `LICENSE` | MIT License |

---

## Reproduce the main result

```bash
python3 bench_FT_vs_FFT.py

Expected output (Apple M1):
FFT  ~0.866 ms
FTT  ~0.530 ms

----------------------------------------------------------------------------------

Scientific scope

This work is a proposal toward a new computational class:

fractal spectral operators — generalizing the FFT.

Immediate research directions:
	•	2D / 3D transforms (images, video, tensor fields)
	•	CNN spectral layers → FTT as drop–in replacement
	•	graph spectral filtering (Cora / Citeseer)
	•	GPU kernels (CUDA / Metal) for scaling limits

⸻

Citation

If you use this in research, cite the PDF in this repository:

Zeinel, M. O. (2025). Fractal–Topological Transform (FTT): A near–linear–time transform for fast convolution, correlation, and graph filtering. GitHub Repository.

⸻

License

MIT — open research, reproducible science.

⸻

Contact

Author: Mohamed Orhan Zeinel
Email: mohamedorhanzeinel@gmail.com

Collaboration for benchmarking, reproducibility studies, and research partnerships is welcome.
