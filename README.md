# fp4-on-sm86: Reproducing & Benchmarking HINT4 (FP4/INT4 via INT8 Tensor Cores) on Ampere `sm_86`

Independent reproduction, PTX/SASS instruction census across 6 NVIDIA GPU generations (`sm_80`–`sm_120`), `Qwen3-0.6B` numerical evaluation, and micro-benchmark breakdown of **HINT4** (*"Towards a hardware-invariant NVFP4 for GPUs without FP4: progress report 1"* by `secemp`, Oct 2026) on **NVIDIA GeForce RTX 3080 Laptop GPU (`sm_86`)**.

---

## 1. Repository Contents

| File | Description |
|---|---|
| [`exp1_numerics_and_census.py`](exp1_numerics_and_census.py) | ① PTX $\to$ SASS instruction census across `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, `sm_120` (`mma.s4` vs `mma.s8`); ② Dual-Grid (`FP4×2 = {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}` vs symmetric `INT4 {-7..+7}` with grid flag in `E4M3` sign bit) & Two-Level (`TL`, `K=32` shared `E4M3` scale with `1..8` integer sub-multipliers); ③ Kurtosis & Hadamard rotation analysis on 28-layer `Qwen/Qwen3-0.6B` and synthetic outlier tensors; ④ End-to-end perplexity (PPL) and KL divergence with `W4A8` and `o_proj` `W8` protection |
| [`exp1_numerics_and_census.json`](exp1_numerics_and_census.json) | Exact JSON output from `exp1_numerics_and_census.py` on RTX 3080 Laptop GPU |
| [`exp2_speed_paradox.py`](exp2_speed_paradox.py) | Micro-benchmark across `M ∈ {1, 8, 16, 32, 64, 256, 1024}` for `[N=4096, K=4096]` and `[N=11008, K=4096]` comparing `FP16 cuBLAS`, `4-bit GEMV`, `Triton W4A16`, `Triton W4A8` (with & without dynamic per-token INT8 activation quantization), and `cuBLAS INT8` — resolving the author's open question on why `M=8` is slower than `FP16` while `M=1` is ~2× faster |
| [`exp2_speed_paradox.json`](exp2_speed_paradox.json) | Exact JSON timings (in microseconds) from `exp2_speed_paradox.py` on RTX 3080 Laptop GPU |
| [`gen_svgs.py`](gen_svgs.py) | Script generating the 3 architecture & benchmark SVG diagrams (`sass-s4-vs-s8-census.svg`, `dual-grid-tl-mechanism.svg`, `m8-speed-paradox-breakdown.svg`) |

---

## 2. Key Experimental Findings

### (1) PTX `.s4` vs `.s8` SASS Census (`sm_80` through `sm_120`)

Compiling a single `mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32` vs `.s8.s8.s32` via `ptxas` and inspecting the machine code via `cuobjdump -sass`:

| Target Architecture | PTX `.s4` (4-bit Signed Int) SASS Output | PTX `.s8` (8-bit Signed Int) SASS Output |
|---|---|---|
| `sm_80` (A100) | `1` × `IMMA.16832.S4.S4` (`0` LOP3, `0` SHF) | `1` × `IMMA.16832.S8.S8` (`0` LOP3, `0` SHF) |
| `sm_86` (RTX 30 Series) | `1` × `IMMA.16832.S4.S4` (`0` LOP3, `0` SHF) | `1` × `IMMA.16832.S8.S8` (`0` LOP3, `0` SHF) |
| `sm_89` (RTX 40 Series) | `1` × `IMMA.16832.S4.S4` (`0` LOP3, `0` SHF) | `1` × `IMMA.16832.S8.S8` (`0` LOP3, `0` SHF) |
| `sm_90` (H100) | **`2` × `IMMA.16816.S8.S8` + `45` LOP3 + `24` SHF (71 insns)** | `1` × `IMMA.16832.S8.S8` (`0` LOP3, `0` SHF) |
| `sm_100` (B200) | **`2` × `IMMA.16816.S8.S8` + `45` LOP3 + `24` SHF (71 insns)** | `1` × `IMMA.16832.S8.S8` (`0` LOP3, `0` SHF) |
| `sm_120` (RTX 50 Series) | **`2` × `IMMA.16816.S8.S8` + `45` LOP3 + `24` SHF (71 insns)** | `1` × `IMMA.16832.S8.S8` (`0` LOP3, `0` SHF) |

### (2) `Qwen3-0.6B` Numerics: Dual-Grid, Two-Level Scaling & Rotation Paradox

| Quantization Configuration | 35-Matrix Mean SNR (↑) | Whole-Model KL vs BF16 (↓) | Whole-Model PPL (↓) |
|---|---:|---:|---:|
| `BF16` Baseline | `∞` | `0.00` | `56.55` |
| `NVFP4` Single-Grid (`W4A16`) | `20.46 dB` | `26.05` | `68.09` |
| `INT4` Symmetric Single-Grid (`W4A16`) | `21.14 dB` | `30.64` | `65.24` |
| `INT4` + Random Hadamard Rotation (`W4A16`) | `21.29 dB` | `34.92` | `71.47` (degraded) |
| `HINT4` Dual-Grid + Scale Search (`W4A16`) | **`22.44 dB`** (FP4 wins `42.3%` blocks) | **`21.73`** | `69.56` |
| `HINT4 TL` Two-Level `K=32` (`W4A8`) | `21.73 dB` | `29.69` | `65.34` |
| **`HINT4 TL` (`W4A8`) + `o_proj` kept in `W8`** | **`21.73 dB`** | **`27.18`** | **`64.46` (best)** |

### (3) Resolving the `M=8` Speed Paradox on RTX 3080 Laptop (`sm_86`)

| Shape & Batch `M` | `FP16 cuBLAS` | `4-bit GEMV` | Dynamic `ActQuant` (FP16$\to$INT8) | `W4A8` Kernel Only | `W4A8` Total (vs `FP16`) |
|---|---:|---:|---:|---:|---:|
| `[4096, 4096]` `M=1` | `84.99 μs` | **`83.97 μs`** | N/A | `105.47 μs` | `83.97 μs` (`1.01×`) |
| `[4096, 4096]` `M=8` | **`96.26 μs`** | — | **`119.81 μs`** | `102.40 μs` | `321.54 μs` (**`0.30×`**) |
| `[11008, 4096]` `M=1` | `238.59 μs` | **`132.10 μs`** | N/A | `160.77 μs` | `132.10 μs` (**`1.81×`**) |
| `[11008, 4096]` `M=8` | `228.35 μs` | — | **`139.26 μs`** | `154.62 μs` | `189.44 μs` (`1.21×`) |
| `[11008, 4096]` `M=32` | **`223.23 μs`** | — | **`111.62 μs`** | `177.15 μs` | `226.30 μs` (**`0.99×`**) |

---

## 3. Running the Experiments

```bash
python3 exp1_numerics_and_census.py
python3 exp2_speed_paradox.py
```
