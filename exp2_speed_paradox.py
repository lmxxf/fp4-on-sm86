#!/usr/bin/env python3
"""
Experiment 2 for WeChat Issue 357:
Evaluating a custom Dual-Grid + Two-Level W4A8 Triton prototype, a verified M=1 Group-16 INT4 GEMV baseline,
and Native cuBLAS INT8 vs FP16 cuBLAS on RTX 3080 Laptop (sm_86):
1. `simple_w4_m1_gemv_kernel`:
   - Encodes w_orig into its own dedicated group-16 symmetric INT4 packed tensor (`w_packed_i4`, `scale16_fp16`, `w_deq_i4_fp16`)
   - Verifies both cosine similarity (> 0.9999) and relative L2 error (< 0.005) against `x_fp16 @ w_deq_i4_fp16.T`
2. `hint4_tl_w4a8_gemm_kernel` (Custom 5.5-bit Dual-Grid + Two-Level W4A8 Prototype):
   - Encodes w_orig into `w_packed_tl` (2 nibbles/byte), `meta_u8` (flag + multiplier m in 1..8), and E4M3-quantized `scale32`
   - Pairs with a fused Triton dynamic INT8 activation quantizer (`act_quant_triton_kernel`)
   - Strictly asserts `cosine_sim > 0.9999` and `rel_l2_err < 0.005` against exact reference `(x_q * x_s) @ w_deq_tl.T`
3. Also records `torch._int_mm` (cuBLAS INT8) in the same script & JSON for traceable upper-bound comparison.
"""
import json
from pathlib import Path
import torch
import triton
import triton.language as tl

OUT_DIR = Path(__file__).resolve().parent
DEVICE = "cuda"


@triton.jit
def act_quant_triton_kernel(
    x_ptr, out_q_ptr, out_s_ptr,
    K,
    BLOCK_K: tl.constexpr
):
    """Fused single-kernel per-row dynamic INT8 activation quantization."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    x = tl.load(x_ptr + row * K + offs).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    amax = tl.maximum(amax, 1e-6)
    scale = amax / 127.0
    q = tl.extra.cuda.libdevice.rint(x / scale)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    tl.store(out_q_ptr + row * K + offs, q)
    tl.store(out_s_ptr + row, scale)


@triton.jit
def simple_w4_m1_gemv_kernel(
    x_ptr, w_packed_ptr, scale16_ptr, out_ptr,
    N, K,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """M=1 4-bit GEMV reading dedicated group-16 INT4 packed weights + FP16 group-16 scales (5.0 bit/weight resident)."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k_half = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
        w_u8 = tl.load(w_packed_ptr + offs_n[:, None] * (K // 2) + offs_k_half[None, :])
        lo = (w_u8 & 0x0F).to(tl.int8) - 7
        hi = (w_u8 >> 4).to(tl.int8) - 7

        x_lo = tl.load(x_ptr + k0 + tl.arange(0, BLOCK_K // 2) * 2).to(tl.float32)
        x_hi = tl.load(x_ptr + k0 + tl.arange(0, BLOCK_K // 2) * 2 + 1).to(tl.float32)

        scale_idx = (k0 // 16) + (tl.arange(0, BLOCK_K // 2) * 2) // 16
        s = tl.load(scale16_ptr + offs_n[:, None] * (K // 16) + scale_idx[None, :]).to(tl.float32)

        acc += tl.sum((lo.to(tl.float32) * x_lo[None, :] + hi.to(tl.float32) * x_hi[None, :]) * s, axis=1)

    tl.store(out_ptr + offs_n, acc.to(tl.float16))


@triton.jit
def hint4_tl_w4a8_gemm_kernel(
    x_int8_ptr, x_scale_ptr, w_packed_ptr, meta_ptr, w_scale32_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Custom Dual-Grid + Two-Level W4A8 GEMM prototype (BLOCK_K=32 matching mma.m16n8k32.s8):
    1. Loads packed 4-bit weights [BLOCK_K=32, BLOCK_N]
    2. Decodes Dual-Grid (INT4 vs FP4*2 via meta bit 7) and multiplies by sub-block multiplier m in {1..8} -> exact INT8
    3. Multiplies via INT8 Tensor Core (tl.dot(int8, int8) -> int32)
    4. Converts INT32 accumulator to FP32 and multiplies by shared 32-weight scale w_s32
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc_f32 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_i8 = tl.load(x_int8_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, mask=(offs_m[:, None] < M), other=0)

        offs_k_half = (k0 // 2) + (tl.arange(0, BLOCK_K) // 2)
        shift = (tl.arange(0, BLOCK_K) & 1) * 4
        w_u8 = tl.load(w_packed_ptr + offs_n[None, :] * stride_wn + offs_k_half[:, None] * stride_wk)
        nib = ((w_u8 >> shift[:, None]) & 0x0F).to(tl.int8)

        sub_in_tile = tl.arange(0, BLOCK_K) // 16
        sub_global = (k0 // 16) + sub_in_tile
        meta_u8 = tl.load(meta_ptr + offs_n[None, :] * (K // 16) + sub_global[:, None])
        is_fp4 = (meta_u8 >> 7) & 1
        mult = (meta_u8 & 0x0F).to(tl.int8)

        q_i4 = nib - 7
        mag = nib & 7
        fp4_abs = mag + tl.where(mag > 4, mag - 4, 0) + tl.where(mag == 7, 2, 0)
        q_fp4 = tl.where((nib & 8) != 0, -fp4_abs, fp4_abs)
        b_i8 = (tl.where(is_fp4 != 0, q_fp4, q_i4) * mult).to(tl.int8)

        dot_i32 = tl.dot(a_i8, b_i8, out_dtype=tl.int32)
        w_s = tl.load(w_scale32_ptr + offs_n * (K // 32) + (k0 // 32))
        acc_f32 += dot_i32.to(tl.float32) * w_s[None, :]

    x_s = tl.load(x_scale_ptr + offs_m, mask=(offs_m < M), other=1.0)
    out = (acc_f32 * x_s[:, None]).to(tl.float16)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], out, mask=(offs_m[:, None] < M))


def encode_group16_int4_weights(w_fp16: torch.Tensor):
    """Dedicated encoder for the M=1 group-16 INT4 GEMV baseline."""
    N, K = w_fp16.shape
    wf = w_fp16.float().reshape(N, K // 16, 16)
    absmax = wf.abs().amax(dim=-1).clamp(min=1e-6)
    scale16_fp16 = (absmax / 7.0).to(torch.float16).contiguous()
    s_f32 = scale16_fp16.float().unsqueeze(-1)
    q = torch.clamp(torch.round(wf / s_f32), -7.0, 7.0)
    codes = (q + 7.0).to(torch.uint8).reshape(N, K)
    w_packed_i4 = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    w_deq_i4 = (q * s_f32).reshape(N, K).to(torch.float16).contiguous()
    return w_packed_i4, scale16_fp16, w_deq_i4


def encode_hint4_tl_weights(w_fp16: torch.Tensor):
    """
    Packs an FP16 weight matrix [N, K] into our custom 5.5-bit Dual-Grid + Two-Level format
    (with E4M3-quantized shared scale per 32 weights, pre-decoded to FP32 for runtime):
      - w_packed_u8: [N, K // 2] uint8 (2 4-bit codes per byte)
      - meta_u8: [N, K // 16] uint8 (bit 7 = is_fp4 flag, bits 0..3 = multiplier m in 1..8)
      - scale32_fp32: [N, K // 32] float32 (E4M3-constrained shared scale per 32 weights)
      - w_dequant_fp32: [N, K] float32 exact dequantized reference weights
    """
    N, K = w_fp16.shape
    wf = w_fp16.float().reshape(N, K // 32, 2, 16)
    pair_absmax = wf.abs().amax(dim=(2, 3)).clamp(min=1e-6)
    global_s = max((pair_absmax.max().item() / 56.0) / 224.0, 1e-12)
    raw_s = torch.clamp((pair_absmax / 56.0) / global_s, min=2**-9, max=448.0)
    scale32 = (raw_s.to(torch.float8_e4m3fn).to(torch.float32) * global_s).contiguous()

    fp4_mags = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0], device=w_fp16.device)
    codes = torch.zeros((N, K // 32, 2, 16), dtype=torch.uint8, device=w_fp16.device)
    metas = torch.zeros((N, K // 32, 2), dtype=torch.uint8, device=w_fp16.device)
    w_deq = torch.zeros_like(wf)

    for sub in range(2):
        w_sub = wf[:, :, sub, :]
        best_err = torch.full((N, K // 32, 1), float("inf"), device=w_fp16.device)
        best_code = torch.zeros((N, K // 32, 16), dtype=torch.uint8, device=w_fp16.device)
        best_meta = torch.zeros((N, K // 32, 1), dtype=torch.uint8, device=w_fp16.device)
        best_recon = torch.zeros_like(w_sub)

        for m in range(1, 9):
            s_eff = (scale32 * float(m)).unsqueeze(-1)
            q_i4 = torch.clamp(torch.round(w_sub / s_eff), -7.0, 7.0)
            c_i4 = (q_i4 + 7.0).to(torch.uint8)
            r_i4 = (q_i4 * float(m)) * scale32.unsqueeze(-1)
            e_i4 = ((w_sub - r_i4) ** 2).sum(dim=-1, keepdim=True)
            better = e_i4 < best_err
            best_err = torch.where(better, e_i4, best_err)
            best_code = torch.where(better, c_i4, best_code)
            best_meta = torch.where(better, torch.full_like(best_meta, m), best_meta)
            best_recon = torch.where(better, r_i4, best_recon)

            norm_abs = (w_sub / s_eff).abs()
            mag_idx = (norm_abs.unsqueeze(-1) - fp4_mags).abs().argmin(dim=-1)
            q_abs = fp4_mags[mag_idx]
            sign_bit = (w_sub < 0) & (mag_idx > 0)
            q_fp4 = torch.where(sign_bit, -q_abs, q_abs)
            c_fp4 = (mag_idx.to(torch.uint8) | (sign_bit.to(torch.uint8) << 3))
            r_fp4 = (q_fp4 * float(m)) * scale32.unsqueeze(-1)
            e_fp4 = ((w_sub - r_fp4) ** 2).sum(dim=-1, keepdim=True)
            better_f = e_fp4 < best_err
            best_err = torch.where(better_f, e_fp4, best_err)
            best_code = torch.where(better_f, c_fp4, best_code)
            best_meta = torch.where(better_f, torch.full_like(best_meta, (1 << 7) | m), best_meta)
            best_recon = torch.where(better_f, r_fp4, best_recon)

        codes[:, :, sub, :] = best_code
        metas[:, :, sub] = best_meta.squeeze(-1)
        w_deq[:, :, sub, :] = best_recon

    codes_flat = codes.reshape(N, K)
    w_packed = (codes_flat[:, 0::2] | (codes_flat[:, 1::2] << 4)).contiguous()
    meta_flat = metas.reshape(N, K // 16).contiguous()
    return w_packed, meta_flat, scale32, w_deq.reshape(N, K).contiguous()


def verify_close(actual: torch.Tensor, expected: torch.Tensor, label: str, max_rel_l2: float = 0.005):
    assert torch.isfinite(actual).all(), f"{label}: non-finite values in output!"
    a = actual.float().reshape(-1)
    e = expected.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a, e, dim=0).item()
    rel_l2 = (torch.norm(a - e) / torch.norm(e).clamp(min=1e-12)).item()
    assert cos > 0.9999 and rel_l2 < max_rel_l2, f"{label} failed verification: cos={cos:.6f}, rel_l2={rel_l2:.6f}"
    return round(cos, 6), round(rel_l2, 6)


def bench_fn(fn, warmup=25, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(start_events, end_events)]
    times_us.sort()
    return round(times_us[len(times_us) // 2], 2)


def bench_two_stage(stage1_fn, stage2_fn, warmup=25, rep=100):
    for _ in range(warmup):
        stage1_fn()
        stage2_fn()
    torch.cuda.synchronize()
    e0 = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    e1 = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    e2 = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        e0[i].record()
        stage1_fn()
        e1[i].record()
        stage2_fn()
        e2[i].record()
    torch.cuda.synchronize()
    runs = []
    for i in range(rep):
        t1 = e0[i].elapsed_time(e1[i]) * 1000.0
        t2 = e1[i].elapsed_time(e2[i]) * 1000.0
        runs.append((t1 + t2, t1, t2))
    runs.sort(key=lambda x: x[0])
    _, med_t1, med_t2 = runs[len(runs) // 2]
    r_t1 = round(med_t1, 2)
    r_t2 = round(med_t2, 2)
    r_tot = round(r_t1 + r_t2, 2)
    return r_t1, r_t2, r_tot


def main():
    torch.manual_seed(0)
    shapes = [(4096, 4096), (11008, 4096)]
    m_list = [1, 8, 16, 32, 64, 256, 1024]
    results = {}

    for N, K in shapes:
        shape_key = f"N{N}_K{K}"
        results[shape_key] = {}
        w_orig = (torch.randn(N, K, dtype=torch.float16, device=DEVICE) * 0.02).contiguous()

        # Dedicated encoding for M=1 Group-16 INT4 GEMV baseline
        w_packed_i4, scale16_fp16, w_deq_i4_fp16 = encode_group16_int4_weights(w_orig)
        # Encoding for Custom Dual-Grid + Two-Level W4A8 Prototype
        w_packed_tl, meta_u8, scale32, w_deq_tl_f32 = encode_hint4_tl_weights(w_orig)
        w_deq_tl_fp16 = w_deq_tl_f32.to(torch.float16).contiguous()
        w_int8_col = torch.randint(-120, 120, (N, K), dtype=torch.int8, device=DEVICE).T

        fp16_bytes = N * K * 2
        simple_w4_bytes = w_packed_i4.numel() + scale16_fp16.numel() * 2  # 5.0 bit/weight
        hint4_proto_bytes = w_packed_tl.numel() + meta_u8.numel() + scale32.numel() * 4  # 5.5 bit/weight

        for M in m_list:
            x_fp16 = (torch.randn(M, K, dtype=torch.float16, device=DEVICE) * 0.5).contiguous()
            out_buf = torch.empty((M, N), dtype=torch.float16, device=DEVICE)
            x_q_buf = torch.empty((M, K), dtype=torch.int8, device=DEVICE)
            x_s_buf = torch.empty((M,), dtype=torch.float32, device=DEVICE)

            # 1. FP16 cuBLAS baseline
            t_fp16_cublas = bench_fn(lambda: torch.matmul(x_fp16, w_deq_tl_fp16.T))

            # Fused Triton ActQuant step
            def act_quant_step():
                act_quant_triton_kernel[(M,)](x_fp16, x_q_buf, x_s_buf, K, BLOCK_K=K)

            def act_quant_eager():
                amax = x_fp16.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                scale_x = amax / 127.0
                x_q_buf.copy_(torch.clamp(torch.round(x_fp16 / scale_x), -127, 127).to(torch.int8))
                x_s_buf.copy_(scale_x.squeeze(-1).to(torch.float32))

            t_act_quant_eager = bench_fn(act_quant_eager)

            # 2. M=1 Decode GEMV (verified against its own group-16 INT4 dequantized reference!)
            if M == 1:
                grid_gemv = (N // 64,)
                out_gemv = torch.empty((M, N), dtype=torch.float16, device=DEVICE)
                simple_w4_m1_gemv_kernel[grid_gemv](x_fp16, w_packed_i4, scale16_fp16, out_gemv, N, K, BLOCK_N=64, BLOCK_K=128)
                torch.cuda.synchronize()
                ref_i4 = torch.matmul(x_fp16.float(), w_deq_i4_fp16.float().T)
                cos_m1_i4, rel_l2_m1_i4 = verify_close(out_gemv, ref_i4, f"{shape_key} M=1 simple_w4_gemv")
                t_w4_gemv = bench_fn(lambda: simple_w4_m1_gemv_kernel[grid_gemv](
                    x_fp16, w_packed_i4, scale16_fp16, out_gemv, N, K, BLOCK_N=64, BLOCK_K=128
                ))
            else:
                t_w4_gemv = None
                cos_m1_i4 = None
                rel_l2_m1_i4 = None

            # 3. Two-stage in-flight timing for Dual-Grid + TL W4A8 Prototype
            BLOCK_M = 16 if M <= 16 else (32 if M <= 32 else 64)
            BLOCK_N = 64
            BLOCK_K = 32
            grid_2d = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

            def kernel_step():
                hint4_tl_w4a8_gemm_kernel[grid_2d](
                    x_q_buf, x_s_buf, w_packed_tl, meta_u8, scale32, out_buf,
                    M, N, K, x_q_buf.stride(0), x_q_buf.stride(1), w_packed_tl.stride(0), w_packed_tl.stride(1),
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
                )

            act_quant_step()
            kernel_step()
            torch.cuda.synchronize()
            ref_w4a8 = torch.matmul((x_q_buf.float() * x_s_buf.unsqueeze(-1)), w_deq_tl_f32.T)
            cos_w4a8, rel_l2_w4a8 = verify_close(out_buf, ref_w4a8, f"{shape_key} M={M} hint4_tl_w4a8")

            t_act_triton, t_w4a8_kernel, t_w4a8_total = bench_two_stage(act_quant_step, kernel_step)

            # 4. Native cuBLAS INT8 (torch._int_mm, pure W8A8 hardware reference)
            M_pad = max(M, 24)
            if M_pad % 8 != 0:
                M_pad = ((M_pad + 7) // 8) * 8
            x_int8_pad = torch.randint(-120, 120, (M_pad, K), dtype=torch.int8, device=DEVICE)
            t_cublas_int8 = bench_fn(lambda: torch._int_mm(x_int8_pad, w_int8_col))

            results[shape_key][f"M={M}"] = {
                "fp16_cublas_us": t_fp16_cublas,
                "simple_w4_gemv_m1_us": t_w4_gemv,
                "speedup_simple_w4_gemv_m1_vs_fp16": round(t_fp16_cublas / t_w4_gemv, 2) if t_w4_gemv else None,
                "m1_gemv_cos_vs_ref": cos_m1_i4,
                "m1_gemv_rel_l2_err": rel_l2_m1_i4,
                "act_quant_triton_us": t_act_triton,
                "act_quant_eager_pytorch_us": t_act_quant_eager,
                "hint4_w4a8_kernel_only_us": t_w4a8_kernel,
                "hint4_w4a8_total_us": t_w4a8_total,
                "speedup_hint4_w4a8_vs_fp16": round(t_fp16_cublas / t_w4a8_total, 2),
                "w4a8_cos_vs_ref": cos_w4a8,
                "w4a8_rel_l2_err": rel_l2_w4a8,
                "cublas_int8_w8a8_us": t_cublas_int8,
                "speedup_cublas_int8_vs_fp16": round(t_fp16_cublas / t_cublas_int8, 2),
                "fp16_weight_mb": round(fp16_bytes / 1e6, 2),
                "simple_w4_resident_mb": round(simple_w4_bytes / 1e6, 2),
                "hint4_proto_resident_mb": round(hint4_proto_bytes / 1e6, 2),
            }
            print(
                f"{shape_key} M={M:4d} | FP16={t_fp16_cublas:6.2f}us | "
                f"M1_W4_GEMV={str(t_w4_gemv):>6} (relL2={str(rel_l2_m1_i4):>8}) | ActQ={t_act_triton:5.2f}us + "
                f"W4A8_Kernel={t_w4a8_kernel:6.2f}us = Total={t_w4a8_total:6.2f}us ({results[shape_key][f'M={M}']['speedup_hint4_w4a8_vs_fp16']}x, relL2={rel_l2_w4a8:.6f}) | "
                f"cuBLAS_INT8={t_cublas_int8:6.2f}us"
            )

    (OUT_DIR / "exp2_speed_paradox.json").write_text(json.dumps(results, indent=2) + "\n")
    print("\nSaved to exp2_speed_paradox.json")


if __name__ == "__main__":
    main()
