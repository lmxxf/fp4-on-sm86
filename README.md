# fp4-on-sm86

[English](#fp4-on-sm86) | **[中文说明](./README_zh.md)**

Independent mechanistic analysis, PTX/SASS disassembly census, numerical fake-quantization evaluation, and Triton W4A8/GEMV kernel benchmarks for `secemp`'s **HINT4** (*Towards a hardware-invariant NVFP4 for GPUs without FP4: progress report 1*, 2026-10-13), executed on an **NVIDIA GeForce RTX 3080 Laptop GPU (`sm_86`, Ampere)**.

## Files

- [`exp1_numerics_and_census.py`](./exp1_numerics_and_census.py) / [`exp1_numerics_and_census.json`](./exp1_numerics_and_census.json):
  1. **6-Architecture PTX $\to$ SASS Census (`sm_80` to `sm_120`, `ptxas` CUDA 12.8 `V12.8.93`, PTX `.version 8.7`)**: Compiles `mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32` vs `.s8.s8.s32` across `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, and `sm_120`. Confirms that `sm_80/86/89` emit `1` native `IMMA.16832.S4.S4` (`0` LOP3, `0` SHF), whereas `sm_90/100/120` expand the `.s4` MMA into `2` `IMMA.16816.S8.S8` + `45` `LOP3` + `24` `SHF` (`71` related SASS instructions), while `.s8` compiles to `1` native `IMMA.16832.S8.S8` across all 6 targets.
  2. **`Qwen/Qwen3-0.6B` Weight Kurtosis & Reconstruction SNR (35 Matrices across 5 Layers)**: Evaluates single-grid `NVFP4`, single-grid symmetric `INT4` (with and without scale search), `HINT4` Dual-Grid (`INT4` vs `FP4*2 = {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}` sharing a unified per-tensor scale), and a Two-Level (`TL`, `K=32` shared E4M3 scale + `m in 1..8` integer sub-block multipliers) prototype, plus short-text BF16 fake-quantization perplexity and per-token KL divergence.
- [`exp2_speed_paradox.py`](./exp2_speed_paradox.py) / [`exp2_speed_paradox.json`](./exp2_speed_paradox.json):
  - Implements a custom 5.5-bit Dual-Grid + Two-Level `W4A8` Triton prototype (packing 2 nibbles per byte + per-subblock dual-grid flag & multiplier `m in 1..8` + E4M3-quantized shared scale per 32 weights pre-decoded to FP32), paired with a fused Triton dynamic INT8 activation quantizer, a dedicated group-16 INT4 `M=1` GEMV baseline, and `torch._int_mm` (`cuBLAS INT8`), with strict `cosine_sim > 0.9999` and `rel_l2_err < 0.005` verification on both `M=1` and `W4A8` paths.
- [`gen_svgs.py`](./gen_svgs.py): Generates `dual-grid-tl-mechanism.svg`, `sass-s4-vs-s8-census.svg`, and `m8-speed-paradox-breakdown.svg`.

## Key Results Summary

### 1. PTX `.s4` vs `.s8` SASS Disassembly Census (`ptxas` V12.8.93)

| Target Architecture | `mma.m16n8k32.s4` SASS Output | `mma.m16n8k32.s8` SASS Output |
|---|---|---|
| `sm_80` / `sm_86` / `sm_89` | `1` `IMMA.16832.S4.S4` (`0` LOP3, `0` SHF) | `1` `IMMA.16832.S8.S8` (`0` LOP3, `0` SHF) |
| `sm_90` / `sm_100` / `sm_120` | **`2` `IMMA.16816.S8.S8` + `45` LOP3 + `24` SHF** | `1` `IMMA.16832.S8.S8` (`0` LOP3, `0` SHF) |

### 2. `Qwen3-0.6B` Reconstruction SNR (35 Matrices) & Short-Text Fake-Quant Eval (`T=152` tokens)

| Scheme | 35-Matrix Mean SNR (dB) | Mean Per-Token KL | Short-Text PPL (BF16 Fake-Quant) |
|---|---:|---:|---:|
| `BF16` Baseline | $\infty$ | `0.0000` | `56.55` |
| `NVFP4` Single-Grid (R2N / Scale Search) | `20.46` / `21.31 dB` | R2N: `0.1714` | R2N: `68.09` |
| `INT4` Single-Grid (R2N / Scale Search) | `21.14` / `21.53 dB` | `0.2016` / `0.2004` | **`65.24`** / `67.62` |
| `INT4` + Hadamard Rotation (`W4A16` Sim) | `21.29 dB` | `0.2297` | `71.47` |
| `HINT4` Dual-Grid (R2N / Scale Search, `W4A16` Sim) | `21.44` / **`22.44 dB`** | Search: **`0.1399`** | Search: `67.59` |
| `HINT4` Dual-Grid (`W4A8` Sim) | `22.44 dB` | `0.1796` | `67.25` |
| `HINT4 TL` Prototype (`K=32`, `W4A8` Sim) | `21.73 dB` | `0.2046` | `72.43` |
| `HINT4 TL` Prototype + `o_proj` Kept at `W8` | **`25.16 dB`** | `0.1911` | `71.20` |

### 3. RTX 3080 Laptop (`sm_86`) Split Timings (`[N=11008, K=4096]`)

| Batch `M` | `FP16 cuBLAS` | Group-16 `4-bit GEMV` (`M=1`) | Col 4: `W4A8` ActQuant | Col 5: Dual-Grid+TL `W4A8` Kernel | Col 6: `W4A8` Total (`4+5`, vs `FP16`) | `cuBLAS INT8` (`W8A8`) | `rel_l2_err` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M=1` | `230.40 μs` | **`132.10 μs` (`1.74×`)** | `9.22 μs` | `321.54 μs` | `330.76 μs` (`0.70×`) | `115.71 μs` | `0.000206` |
| `M=8` | **`228.35 μs`** | — | `6.14 μs` | `315.39 μs` | `321.53 μs` (`0.71×`) | `115.71 μs` | `0.000208` |
| `M=16` | **`230.40 μs`** | — | `5.12 μs` | `332.80 μs` | `337.92 μs` (`0.68×`) | `115.71 μs` | `0.000207` |
| `M=32` | **`225.28 μs`** | — | `8.19 μs` | `330.75 μs` | `338.94 μs` (`0.66×`) | `116.74 μs` | `0.000208` |
| `M=256` | **`637.95 μs`** | — | `12.29 μs` | `1591.30 μs` | `1603.59 μs` (`0.40×`) | `224.26 μs` (`2.84×`) | `0.000207` |
| `M=1024` | **`2617.34 μs`** | — | `31.74 μs` | `6345.73 μs` | `6377.47 μs` (`0.41×`) | `898.05 μs` (`2.91×`) | `0.000207` |

## Reproduction

```bash
python3 exp1_numerics_and_census.py
python3 exp2_speed_paradox.py
python3 gen_svgs.py
```
