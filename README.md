[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17563190.svg)](https://doi.org/10.5281/zenodo.17563190)

# Fractal–Topological Transform (FTT)

**A near–linear–time mathematical transform for fast convolution, correlation, and graph filtering.**

FTT introduces a new algebraic paradigm: combining fractal hierarchical indexing with topology–aware lifting to construct an orthonormal, sparse basis that preserves energy and supports generalized convolution in transform–space.

This work proposes that convolution can be reduced toward **O(N)** for broad families of natural signals — potentially surpassing the classical \(O(N\log N)\) limit of FFT-based methods.

---

## Why this matters

The global compute economy is bottlenecked by convolution cost.

Machine learning, scientific simulation, cryptography, and signal processing are fundamentally limited by complexity bounds.  
FFT broke the \(N^2\) wall.  
FTT is a direct attempt to break the \(N\log N\) wall.

---

## Key Features of FTT

| Property | Summary |
|---------|---------|
| Complexity | Near–linear time per transform level |
| Algebra | New block–structured multiplication in transform domain |
| Structure | Fractal order + topological lifting |
| Variants | Integer (FTT–NTT) and Graph (G–FTT) versions included |
| Reproducibility | Fully included Python reference implementation |

---

## Repository Contents

| File | Purpose |
|------|---------|
| `Fractal-Topological-Transform_FTT.pdf` | Full peer-review-ready scientific manuscript |
| `ftt_reference.py` | Minimal reproducible reference implementation (CPU) |
| `LICENSE` | MIT open-science license |
| `README.md` | This file |

---

## Quick Start

```bash
python ftt_reference.py

This runs synthetic 1D convolution benchmarks and prints median timing results.
Use printed values to populate the experimental tables in the PDF.
---------------------------------------------------------------------------------------------------

Scientific Validation Protocol

The paper includes a complete evaluation protocol (Section “Experimental Protocol”):
	•	Synthetic signals (1D / 2D)
	•	MNIST / CIFAR CNN test (replace first conv layer with FTT)
	•	Graph filtering (Cora / Citeseer)

Metrics:
	•	inference time (ms)
	•	FLOPs estimates
	•	accuracy delta
	•	MSE / PSNR / SSIM
	•	bootstrap CI, Wilcoxon tests

⸻

Citation

If you reference this work, cite the PDF in this repository.
Zeinel, M. O. (2025). Fractal–Topological Transform (FTT): A near–linear–time transform for fast convolution, correlation, and graph filtering. GitHub Repository.

License

MIT License — open research, open science.

⸻

Contact

Author: Mohamed Orhan Zeinel
Email: mohamedorhanzeinel@gmail.com
Collaboration on benchmarking, peer-review, and reproducibility studies is welcome.
