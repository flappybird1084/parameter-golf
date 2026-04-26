# Experiment Log

## Baseline
- File: train_gpt.py (unmodified)
- Hardware: NVIDIA RTX A5000 (single GPU)
- Step time: ~1570ms/step
- Steps completed in 600s: ~382
- val_bpb: 1.5546 (pre-quant), 1.5754 (post-quant int8+zlib=9.2MB)
- Notes: warmdown_iters=1200 covers entire run — LR at step 1 is 0.318 of peak, always decaying

## v1: All Features (2026-04-26)
- File: experiments/v1_optimized.py
- Changes: LeakyReLU^2, VE (ve_scales=ones WRONG), LoRA (rank=16), Recurrence, TTT
- Result: val_bpb=2.5721 — CATASTROPHIC FAILURE
- Root cause: matrix_lr=0.04 with warmdown_iters=115 → LR 6.7x too high; ve_scales=ones added noise from step 1
- Lesson: Cannot use baseline LR with fixed warmdown schedule. Must understand the "broken" LR schedule.

## v2: Fixed LR + Features (2026-04-26)
- File: experiments/v2_fixed_lr.py
- Changes: matrix_lr=0.015, warmdown_iters=180, ve_scales=zeros (fixed), + all v1 features
- Result: val_bpb=1.6464 pre-quant, 1.6529 post-quant, model=10.58MB (under 16MB!)
- Root cause: warmdown_iters=180 creates constant LR phase (0.015 for first 177 steps) which diverges from baseline's smooth decay schedule; total LR energy 89% higher than baseline
- Steps: 359 in 601s, step_avg=1675ms
- Lesson: Baseline's "broken" warmdown_iters=1200 is actually OPTIMAL — creates smooth monotonic LR decay from step 1

## v3: Minimal Improvements + Sliding Window (running 2026-04-26)
- File: experiments/v3_sliding_leaky.py
- Changes from baseline:
  1. LeakyReLU^2 (proven -0.003 BPB in records)
  2. Muon weight decay = 0.04 (from records, helps regularization)
  3. Sliding window eval (EVAL_STRIDE=64, proven -0.032 BPB from SlidingWindowEval record)
- LR schedule: UNCHANGED from baseline (warmdown_iters=1200, same LRs)
- Expected: ~1.514 BPB (from 1.5546 - 0.003 - 0.032 ≈ 1.519, but may vary)
- Status: RUNNING

## v3: Final Result (2026-04-26)
- File: experiments/v3_sliding_leaky.py
- Training val_bpb: 1.5449 (step 381, 600890ms)
- Sliding window eval (stride=64, batch=512): **1.5384 BPB** (int8 roundtrip)
- Model size: 9.0MB int8+zlib (17M params)
- Step time: ~1570ms, 382 steps in 600s
- Notes: Sliding window gave -0.0065 BPB improvement over training eval

## v5: XSA + BigramHash (2026-04-26)
- File: experiments/v5_xsa.py
- Changes from v3: XSA all layers (parameter-free), BigramHash(10240, 128), grad_clip_norm=0.3
- Model size: 18.4M params (1.3M extra for bigram embed+proj), 8.77MB compressed
- Step time: ~1671ms, ~359 steps in 600s
- Training val_bpb: **1.5380** (vs v3's 1.5449 → -0.0069 improvement)
- Post-quant standard eval: 1.5617 (vs v3's post-quant ~1.57)
- Sliding window eval (in progress): trending 1.535-1.537 (vs v3's 1.5384)
- Notes: Step 200 train_loss=2.7618 vs v3's 2.8260 (-0.064 improvement)

## v5 Ablations (2026-04-26)
- v6 (LN Scale + QK=5.0 + partial RoPE + smaller bigram + no clip): 1.6206 — catastrophic failure
  - LN Scale (1/√(layer+1)) too aggressive for 360 steps, last layer at 0.33 scale
  - QK=5.0 overly focused attention from start
- v6b (no LN + no QK=5 + partial RoPE + smaller bigram 2048×112 + no clip + fast Muon warmup): 1.5762 — worse than v5
- v7 (v5 + partial RoPE 16/64 dims only): 1.5525 — worse than v5
- CONCLUSION: v5 is optimal. XSA + BiggramHash(10240,128) + grad_clip=0.3 are all important.
- Partial RoPE hurts: position information important for our short training (360 steps)

## v8: Mini Depth Recurrence — Wrong Layers (2026-04-26)
- File: experiments/v8_recur.py
- Changes from v5: recur_last_n=2 (repeat blocks 7,8 after normal forward)
- Result: val_bpb=1.6351 pre-quant, 1.6718 post-quant — WORSE than v5
- Root cause: wrong layers (7,8 instead of 4,5 U-Net hinge); 296 steps vs 359; step 200 train_loss=2.8732 vs v5's 2.7618
- Lesson: Depth recurrence sweet spot is at U-Net hinge (layers 4,5), NOT last layers

## v9_ema: EMA Model Averaging (2026-04-26)
- File: experiments/v9_ema.py
- Changes from v5: EMA decay=0.98 (lookback=50 steps), EMA model used for final eval
- Result: val_bpb=1.5400 pre-quant, 1.5726 post-quant — WORSE than v5 (1.5617)
- Root cause: smooth warmdown schedule creates monotonic convergence; EMA averages worse early models into final model
- Lesson: EMA hurts when LR decays smoothly from start (no oscillation to average out)

## v8b: Depth Recurrence at Correct Layers 4,5 (2026-04-26)
- File: experiments/v8b_recur.py
- Changes from v5: recur_layers=[4,5], recur_start_step=180 (delayed start), recompile at step 180
- Result: val_bpb=1.5792 pre-quant, 1.6013 post-quant — WORSE than v5, 313 steps
- Root cause: 30s recompile overhead + 22% slower step time = 313 steps vs 359; not enough gradient steps for recurrent layers to converge
- Lesson: Depth recurrence at correct layers 4,5 still hurts at 359-step scale due to compute trade-off

## v12: FP16 Embedding + Weight Snapping QAT (2026-04-26)
- File: experiments/v12_fp16emb.py
- Changes from v5:
  1. FP16 embedding in quantized artifact (tok_emb.weight stored as FP16 instead of int8)
  2. Weight snapping QAT: after optimizer step, snap all matrix weights to int8 grid when LR scale < 0.05 (last ~60 steps)
- Steps: 358, step_avg=1679ms (nearly same as v5)
- Pre-quant training val_bpb: 1.5627 (WORSE than v5's 1.5380 — expected, due to weight snapping)
- Post-quant SLIDING window val_bpb: **1.5320** (vs v5's 1.5344 → **-0.0024 improvement!**)
- Model size: 9.12MB int8+zlib (FP16 embedding adds ~350KB vs v5's 8.77MB)
- Key insight: weight snapping eliminates quantization penalty (model already at int8 grid when quantized)
- v12 is NEW BEST (1.5320 BPB)

## v13: Always-On STE QAT (2026-04-26)
- File: experiments/v13_qat_ste.py
- Changes from v12: STE QAT in CastedLinear.forward() (always active during training), removed weight snapping
- Steps: 355 (4 fewer than v5 due to 18ms STE overhead per step from torch.quantile)
- Pre-quant training val_bpb: 1.5433 (worse than v5's ~1.538)
- Post-quant sliding val_bpb: **1.5356** (vs v5 1.5344, vs v12 1.5320)
- Result: WORSE than v12, slightly worse than v5
- Root cause: Always-on STE adds quantization noise to ALL 355 training steps, hurting early convergence more than it helps quant robustness at our low step count

## Ablation Summary: v12 Breaks the v5 Record
- **v12 (1.5320 BPB) is now the best result**, beating v5 (1.5344 BPB) by 0.0024
- v13 STE QAT (1.5356) is worse than both v5 and v12 — confirms weight snapping is better at 359 steps
- v5 sliding window (1.5344 BPB) was previous best
- Techniques that work at 6000+ steps (8×H100) don't always translate to 359 steps (single A5000):
  - Depth recurrence: requires enough steps to train shared weights effectively
  - EMA: requires oscillating convergence to benefit from averaging
  - Partial RoPE: position info crucial at low step count
  - LN Scale: too aggressive for only 9 layers in 359 steps
  - TTT at eval: too slow (~1.24s/step uncompiled vs 16ms on H100 — 77× slower)

## v16: Muon Momentum Warmup Fix (2026-04-26)
- File: experiments/v16_muon_warmup.py
- Changes from v12: muon_momentum_warmup_steps: 500 → 100
- Root cause fixed: with 500 warmup steps and only 358 total steps, Muon NEVER reached target 0.95 momentum (max 0.922 at step 358). With 100 warmup steps, reaches full 0.95 momentum by step 100 and maintains for 72% of training.
- Steps: 358, step_avg=1676ms, pre-quant val_bpb=1.5433 (vs v12's 1.5627!)
- Post-quant sliding val_bpb: **1.5126** (vs v12's 1.5320 → **-0.0194 improvement!**)
- Model size: 9.63MB (larger due to better-trained weights being less compressible, still fine)
- Step 200 train_loss: 2.7472 (vs v12's 2.7657 → -0.0185 improvement)
- v16 is NEW BEST by huge margin

## v15: FP16 Bigram + 20480 Vocab (2026-04-26)
- File: experiments/v15_fp16bigram.py
- Changes from v12: BIGRAM_VOCAB_SIZE=20480 (2×), FP16 bigram.embed.weight storage, exclude bigram from weight snapping
- Steps: 358, step_avg=1678ms, pre-quant val_bpb=1.5660
- Post-quant sliding val_bpb: **1.5359** (vs v12's 1.5320 → WORSE by 0.0039!)
- Model size: 9.14MB (only 19KB more than v12 despite 2× bigram table — sparse embeds compress to nothing)
- Root cause: 20480 buckets → each gets ~25 gradient updates/step vs ~51 for 10240. Sparser gradients hurt Adam convergence in 359 steps.
- Step 200 loss: 2.7747 (vs v12's 2.7657) — clearly worse from step start
- Lesson: Don't increase bigram vocab beyond 10240 at 359 steps. More buckets = sparser gradients = slower convergence.

## v14: SmearGate (2026-04-26)
- File: experiments/v14_smeargate.py
- Changes from v12: SmearGate (learned per-dim blend of current + previous token embedding, 512 params init sigmoid(3.0)≈0.95)
- Steps: 357, step_avg=1684ms, pre-quant val_bpb=1.5636
- Post-quant sliding val_bpb: **1.5334** (vs v12's 1.5320 → WORSE by 0.0014)
- Conclusion: SmearGate redundant — BigramHash already captures bigram context. Rejected.

## v17: Bigram Dim=256 (2026-04-26)
- File: experiments/v17_bigram256.py
- Changes from v16: BIGRAM_DIM 128 → 256 (extra 1.38M params for wider bigram embedding)
- Model params: 19,812,425 (vs v16's 18,436,169)
- Steps: 358, step_avg=1677ms, step 200 train_loss=2.7518 (vs v16's 2.7472 — slightly worse)
- Post-quant sliding val_bpb: **1.5184** (vs v16's 1.5126 → WORSE by 0.0058)
- Conclusion: Wider bigram dim doesn't help — extra params don't converge in 358 steps

## v18: Quarter Batch (2026-04-26) *** NEW BEST ***
- File: experiments/v18_quarterbatch.py
- Changes from v16: TRAIN_BATCH_TOKENS 524288 → 131072, grad_accum_steps=2 (microbatch = 64 seqs, unchanged)
- Steps: 1377, step_avg=436ms (4x more steps in same 600s!)
- In-training final val_bpb: 1.3833 (step 1377)
- Post-quant sliding val_bpb: **1.3497** (vs v16's 1.5126 → **-0.163 improvement!**)
- Model size: 12.86MB int8+zlib (larger due to better-trained weights, still under 16MB)
- LR schedule: ~177 steps at full LR, then 1200-step warmdown — much better than full-batch always-decaying
- Key insight: smaller batch = more optimizer steps AND better LR schedule. Same gradient quality per microbatch.
- Gap to 1.3 BPB target: only 0.0497

## Notes on 1.3 BPB Target (Updated)
- v18 quarter batch smashed the previous ceiling: 1.5546 → 1.3497 (baseline → best) in one day
- The "notes on 1.3 BPB" estimate of 1.48-1.50 was completely wrong — ignored step count optimization
- Next logical step: eighth batch (65536 tok/step, ~2752 steps) — could reach ~1.30-1.32 BPB
- Risk at eighth batch: gradient noise from 64-seq microbatch may hurt Adam convergence more than extra steps help
- Secondary candidates: add XSA/BigramHash/weight snapping on top of v18 base
